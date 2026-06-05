"""
Performance benchmark on Apple Silicon: median/peak tok/s, TTFT, peak memory.
Runs a fixed prompt N times (after a warmup) and aggregates GenerationResult stats.
"""
import argparse, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--runs", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--prompt", default="Write a detailed technical explanation of how mixed-precision quantization preserves model accuracy. Cover group-wise scaling, the role of the residual stream, and why some layers are kept at higher precision.")
ap.add_argument("--out", default="research/benchmark.json")
a = ap.parse_args()

model, processor = load(a.model)
config = load_config(a.model)
formatted = apply_chat_template(processor, config, a.prompt, num_images=0)

print(f"[bench] warmup")
generate(model, processor, formatted, max_tokens=32, temperature=0.0, verbose=False)

gen_tps, prompt_tps, ttft, peak = [], [], [], []
for i in range(a.runs):
    r = generate(model, processor, formatted, max_tokens=a.max_tokens, temperature=0.0, verbose=False)
    gen_tps.append(r.generation_tps)
    prompt_tps.append(r.prompt_tps)
    ttft.append(r.prompt_tokens / r.prompt_tps if r.prompt_tps else 0.0)
    peak.append(r.peak_memory)
    print(f"  run {i+1}: gen {r.generation_tps:.1f} tok/s | prompt {r.prompt_tps:.0f} tok/s | TTFT {ttft[-1]*1000:.0f} ms | peak {r.peak_memory:.2f} GB | {r.generation_tokens} tok")

res = {
    "model": a.model, "runs": a.runs, "max_tokens": a.max_tokens,
    "gen_tps_median": round(statistics.median(gen_tps), 1),
    "gen_tps_peak": round(max(gen_tps), 1),
    "prompt_tps_median": round(statistics.median(prompt_tps), 0),
    "ttft_ms_median": round(statistics.median(ttft) * 1000, 0),
    "peak_memory_gb": round(max(peak), 2),
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("=== BENCHMARK ===")
print(json.dumps(res, indent=2))
