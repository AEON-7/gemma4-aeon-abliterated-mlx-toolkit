"""
Preflight on the ACTUAL downloaded source, BEFORE the heavy conversion:
  1. Config ground-truth (model_type, PLE on/off, tie, soft-cap, layers, hidden).
  2. Dry-run the recipe predicate over every quantizable Linear and count
     mxfp4 / mxfp8 / bf16-skip WITHOUT quantizing anything.

Expected on the 12B: PLE OFF, and 192 mxfp4 / 136 mxfp8 / 4 skip(bf16).
A deviation here means STOP and re-derive before spending the convert.
"""
import argparse, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import pred  # noqa: E402

import mlx.nn as nn  # noqa: E402
from mlx_vlm.utils import fetch_from_hub, get_model_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf-path", required=True, help="local BF16 source dir (or HF id)")
args = ap.parse_args()

print(f"[preflight] lazy-loading {args.hf_path}")
model, config, processor = fetch_from_hub(get_model_path(args.hf_path), lazy=True, trust_remote_code=True)

tc = config.get("text_config", config)
facts = {
    "model_type": config.get("model_type"),
    "architectures": config.get("architectures"),
    "PLE (hidden_size_per_layer_input)": tc.get("hidden_size_per_layer_input"),
    "tie_word_embeddings": tc.get("tie_word_embeddings"),
    "final_logit_softcapping": tc.get("final_logit_softcapping"),
    "num_hidden_layers": tc.get("num_hidden_layers"),
    "hidden_size": tc.get("hidden_size"),
}
print("=== CONFIG GROUND TRUTH ===")
for k, v in facts.items():
    print(f"  {k:36}: {v}")
ple = tc.get("hidden_size_per_layer_input")
if ple in (0, None):
    print("  -> PLE OFF (good): mixed-FP4 recipe applies as designed.")
else:
    print(f"  -> !!! PLE ON ({ple}): add per_layer/embed_tokens_per_layer to bf16 SKIP and re-verify before converting.")

# Dry-run: walk quantizable Linears, record the recipe's decision, quantize nothing.
counts, ex = Counter(), {}
def counting(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    r = pred(path, module)
    key = "skip(bf16)" if r is False else ("mxfp8" if r["bits"] == 8 else "mxfp4")
    counts[key] += 1
    ex.setdefault(key, []).append(path)
    return False  # never actually quantize during preflight

nn.quantize(model, group_size=32, bits=4, class_predicate=counting)

print("=== PREDICATE DRY-RUN (quantizable Linears only) ===")
for k in ("mxfp4", "mxfp8", "skip(bf16)"):
    print(f"  {k:12}: {counts.get(k, 0):4d}   e.g. {ex.get(k, ['-'])[:2]}")
print(f"  TOTAL quantizable: {sum(counts.values())}")
print("  EXPECTED on 12B  : 192 mxfp4 / 136 mxfp8 / 4 skip(bf16)")
