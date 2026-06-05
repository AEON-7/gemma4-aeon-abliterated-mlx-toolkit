"""
Valid near-lossless metric for an instruct/thinking model: score quants on BF16's
OWN greedy trajectory (the confident regime), not raw-prose teacher forcing.

Phase 1: BF16 greedily generates responses; capture exact token ids + BF16 logprobs
         at each response position.
Phase 2: For each quant, teacher-force the same (prompt+response) sequence and measure
         at response positions:
           - top1_match  : how often the quant's argmax == BF16's actual greedy token
           - mean_kl      : KL(BF16 || quant) in nats (sharp, confident distributions)
A near-lossless quant reproduces BF16's choices (top1_match -> ~1.0, KL -> ~0).
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_vlm import load
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.generate import stream_generate
import mlx.core as mx

BF16 = "out/aeon7-bf16"
QUANTS = {"flagship_8bit": "out/aeon7-mlx8bit", "mixed_fp4": "out/aeon7-mxfp4-mixed"}
PROMPTS = [
    "Explain how a transformer neural network processes a sentence, from tokens to output logits.",
    "Summarize the main causes of the 2008 financial crisis in one paragraph.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Give me a step-by-step beginner plan to start running.",
]
MAXTOK = 140

print("[traj] loading bf16, capturing trajectories")
mb, proc = load(BF16)
config = load_config(BF16)
tok = getattr(proc, "tokenizer", proc)

seqs = []  # (full_ids list, resp_start index)
for p in PROMPTS:
    f = apply_chat_template(proc, config, p, num_images=0)
    prompt_ids = tok.encode(f)
    gen_ids = [int(r.token) for r in stream_generate(mb, proc, f, max_tokens=MAXTOK, temperature=0.0)]
    seqs.append((prompt_ids + gen_ids, len(prompt_ids), gen_ids))

# BF16 logprobs at response positions
bf_store = []
for full, rs, gen in seqs:
    ids = mx.array([full])
    lp = (lambda L: L - mx.logsumexp(L, -1, keepdims=True))(mb(ids).logits[0].astype(mx.float32))
    # positions rs-1 .. len-2 predict tokens rs .. len-1 (== gen_ids)
    sl = lp[rs - 1 : len(full) - 1]
    bf_store.append(np.array(sl.astype(mx.float16)))
del mb
try: mx.clear_cache()
except Exception: pass

results = {}
for label, path in QUANTS.items():
    print(f"[traj] eval {label}")
    mq, _ = load(path)
    tot, match, kls = 0, 0, []
    for (full, rs, gen), bf in zip(seqs, bf_store):
        ids = mx.array([full])
        lq = (lambda L: L - mx.logsumexp(L, -1, keepdims=True))(mq(ids).logits[0].astype(mx.float32))
        slq = lq[rs - 1 : len(full) - 1]
        aq = np.array(slq.argmax(-1))
        gen_arr = np.array(gen[: len(aq)])
        match += int((aq == gen_arr).sum()); tot += len(aq)
        lpb = mx.array(bf.astype(np.float32)); pb = mx.exp(lpb)
        kl = (pb * (lpb - slq)).sum(-1)
        kls.append(np.array(kl))
    del mq
    try: mx.clear_cache()
    except Exception: pass
    kl_all = np.concatenate(kls)
    results[label] = {
        "resp_tokens": tot,
        "top1_match_bf16_greedy": round(match / tot, 4),
        "mean_kl_nats": round(float(kl_all.mean()), 4),
        "median_kl_nats": round(float(np.median(kl_all)), 4),
        "p95_kl_nats": round(float(np.percentile(kl_all, 95)), 4),
    }
    print(json.dumps(results[label], indent=2))

json.dump(results, open("research/traj_quality.json", "w"), indent=2)
print("=== TRAJECTORY QUALITY (on BF16's own greedy output) ===")
print(json.dumps(results, indent=2))
