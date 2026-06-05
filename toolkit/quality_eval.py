"""
Memory-safe quality eval of a quant vs the BF16 source on identical clean text.
Loads ONE model at a time (stores bf16 logits as fp16). Reports:
  ppl_bf16, ppl_quant, ppl_delta_pct   <- the interpretable near-lossless metric
  mean_kl, median_kl (nats)
  top1_all, top1_confident (positions where bf16 max-prob > 0.5)
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_vlm import load
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--bf16", required=True)
ap.add_argument("--quant", required=True)
ap.add_argument("--label", default="quant")
ap.add_argument("--corpus", default="research/clean_corpus.txt")
ap.add_argument("--max-tokens", type=int, default=640)
ap.add_argument("--out", default=None)
a = ap.parse_args()

text = open(a.corpus).read()

print(f"[eval:{a.label}] loading bf16 reference")
mb, proc = load(a.bf16)
tok = getattr(proc, "tokenizer", proc)
ids_list = tok.encode(text)[: a.max_tokens - 1]
bos = getattr(tok, "bos_token_id", 2)
ids_list = [bos] + ids_list
ids = mx.array([ids_list])
N = ids.shape[1]
lb_np = np.array(mb(ids).logits[0].astype(mx.float16))  # (N, V)
del mb
try: mx.clear_cache()
except Exception: pass

print(f"[eval:{a.label}] loading quant {a.quant}")
mq, _ = load(a.quant)
lq = mq(ids).logits[0].astype(mx.float32)

lb = mx.array(lb_np).astype(mx.float32)
lpb = lb - mx.logsumexp(lb, axis=-1, keepdims=True)
lpq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
pb = mx.exp(lpb)
kl = (pb * (lpb - lpq)).sum(axis=-1)                 # (N,)
maxp = pb.max(axis=-1)                                # (N,)
a_b = lb.argmax(-1); a_q = lq.argmax(-1)

tgt = ids[0, 1:N]
idx = mx.arange(N - 1)
nll_b = float((-lpb[idx, tgt]).mean())
nll_q = float((-lpq[idx, tgt]).mean())

kl_np = np.array(kl); maxp_np = np.array(maxp)
agree = np.array(a_b == a_q)
conf = maxp_np > 0.5
res = {
    "label": a.label, "tokens": int(N),
    "ppl_bf16": round(math.exp(nll_b), 3),
    "ppl_quant": round(math.exp(nll_q), 3),
    "ppl_delta_pct": round(100 * (math.exp(nll_q) - math.exp(nll_b)) / math.exp(nll_b), 2),
    "mean_kl_nats": round(float(kl_np.mean()), 4),
    "median_kl_nats": round(float(np.median(kl_np)), 4),
    "top1_all": round(float(agree.mean()), 3),
    "top1_confident": round(float(agree[conf].mean()) if conf.any() else float("nan"), 3),
    "n_confident": int(conf.sum()),
}
print(json.dumps(res, indent=2))
if a.out:
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev[a.label] = res
    json.dump(prev, open(a.out, "w"), indent=2)
