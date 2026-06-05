"""
Post-convert audit: prove the precision map landed in the artifact.
  - .scales present => quantized; only .weight => bf16 (skipped).
  - per-layer quantization block: count mxfp4 vs mxfp8 entries.
  - on-disk size.
Expect: embeds/lm_head/projectors => weight-only (bf16); o_proj/down_proj/v_proj
=> mxfp8; q/k/gate/up => mxfp4; ~192 mxfp4 / 136 mxfp8 per-layer entries.
"""
import argparse, json, os
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--mlx-path", required=True)
d = ap.parse_args().mlx_path

idx_path = os.path.join(d, "model.safetensors.index.json")
keys = []
if os.path.exists(idx_path):
    keys = list(json.load(open(idx_path))["weight_map"].keys())
else:
    try:
        from safetensors import safe_open
        f = [x for x in os.listdir(d) if x.endswith(".safetensors")][0]
        with safe_open(os.path.join(d, f), framework="numpy") as h:
            keys = list(h.keys())
    except Exception as e:
        print("could not read tensor keys:", e)

def has(sub, suf):
    return any(sub in k and k.endswith(suf) for k in keys)

print("=== weight presence (.scales => quantized; only .weight => bf16) ===")
for sub in ["embed_tokens", "lm_head", "embed_vision.embedding_projection",
            "embed_audio.embedding_projection", "vision_embedder",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
    if any(sub in k for k in keys):
        print(f"  {sub:42} weight={has(sub, '.weight')!s:5} scales={has(sub, '.scales')!s:5} biases={has(sub, '.biases')!s:5}")

cfg = json.load(open(os.path.join(d, "config.json")))
q = cfg.get("quantization", {})
print("=== quantization block ===")
print("  global:", {k: q.get(k) for k in ("group_size", "bits", "mode")})
modes = Counter()
for k, v in q.items():
    if isinstance(v, dict):
        modes[v.get("mode", f"affine{v.get('bits')}")] += 1
print("  per-layer override modes:", dict(modes))

tot = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
print(f"=== on-disk safetensors: {tot / 1e9:.2f} GB ===")
