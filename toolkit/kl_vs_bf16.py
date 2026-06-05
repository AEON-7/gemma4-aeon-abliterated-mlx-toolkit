"""
Near-lossless proof: teacher-forced KL(bf16 || fp4) + top-1 agreement + PPL delta
on an identical natural-text sample. Loads both models (memory-managed window loop).
ACCEPT 'below sampling noise' if mean KL < ~1e-3 nats and top-1 agreement > ~99%.
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_vlm import load
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--bf16", required=True)
ap.add_argument("--fp4", required=True)
ap.add_argument("--corpus", default=None, help="text file; default: built-in from source docs")
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--out", default="research/kl_mlxfp4.json")
a = ap.parse_args()

if a.corpus and os.path.exists(a.corpus):
    text = open(a.corpus).read()
else:
    parts = []
    for f in ["out/aeon7-bf16/README.md", "out/aeon7-bf16/AGENTS.md", "out/aeon7-bf16/chat_template.jinja"]:
        if os.path.exists(f):
            parts.append(open(f).read())
    text = "\n\n".join(parts)

print("[kl] loading fp4 (for tokenizer + 2nd model)")
m_fp4, proc = load(a.fp4)
tok = getattr(proc, "tokenizer", proc)
ids_list = tok.encode(text)[: a.max_tokens]
bos = getattr(tok, "bos_token_id", None)
if bos is not None and (not ids_list or ids_list[0] != bos):
    ids_list = [bos] + ids_list[: a.max_tokens - 1]  # Gemma is BOS-sensitive
ids_all = mx.array(ids_list)[None]
N = ids_all.shape[1]
print(f"[kl] corpus tokens: {N} | bos={bos} | first_ids={ids_list[:6]}")

print("[kl] loading bf16 reference (~24GB)")
m_bf16, _ = load(a.bf16)

def clear():
    for fn in ("clear_cache",):
        try:
            getattr(mx, fn)()
        except Exception:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass

W = a.window
sum_kl = 0.0; n_pos = 0; top1 = 0
nll_bf = 0.0; nll_fp = 0.0; nll_n = 0
for s in range(0, N - 1, W):
    e = min(s + W, N)
    ids = ids_all[:, s:e]
    lb = m_bf16(ids).logits.astype(mx.float32)[0]
    lf = m_fp4(ids).logits.astype(mx.float32)[0]
    lpb = lb - mx.logsumexp(lb, axis=-1, keepdims=True)
    lpf = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    pb = mx.exp(lpb)
    kl = (pb * (lpb - lpf)).sum(axis=-1)
    a1 = (lb.argmax(-1) == lf.argmax(-1)).sum()
    mx.eval(kl, a1)
    sum_kl += float(kl.sum()); n_pos += int(kl.shape[0]); top1 += int(a1)
    if e < N:
        tgt = ids_all[0, s + 1 : e + 1]
        idx = mx.arange(lpb.shape[0])
        nb = (-lpb[idx, tgt]).sum(); nf = (-lpf[idx, tgt]).sum()
        mx.eval(nb, nf)
        nll_bf += float(nb); nll_fp += float(nf); nll_n += int(lpb.shape[0])
    del lb, lf, lpb, lpf, pb, kl, a1
    clear()
    print(f"  window {s}-{e}: cum_meanKL={sum_kl/n_pos:.3e} top1={top1/n_pos:.4f}")

res = {
    "corpus_tokens": N, "window": W,
    "mean_kl_nats": round(sum_kl / n_pos, 6),
    "top1_agreement": round(top1 / n_pos, 4),
    "ppl_bf16": round(math.exp(nll_bf / nll_n), 4) if nll_n else None,
    "ppl_fp4": round(math.exp(nll_fp / nll_n), 4) if nll_n else None,
}
res["ppl_delta_pct"] = round(100 * (res["ppl_fp4"] - res["ppl_bf16"]) / res["ppl_bf16"], 4) if nll_n else None
res["verdict_below_noise"] = bool(res["mean_kl_nats"] < 1e-3 and res["top1_agreement"] > 0.99)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("=== KL / PPL RESULT ===")
print(json.dumps(res, indent=2))
