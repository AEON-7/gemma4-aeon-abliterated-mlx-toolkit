"""
PRIMARY BUILD: mixed-precision MLX FP4 for AEON-7 Gemma-4-12B abliterated.
Calls mlx_vlm.convert() with the validated custom callable predicate (recipe.pred).
mxfp4 bulk + mxfp8 residual-writers + bf16 head/projectors. No calibration (RTN).
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import pred  # noqa: E402
from mlx_vlm import convert  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf-path", required=True, help="local BF16 source dir")
ap.add_argument("--mlx-path", required=True, help="output dir for the mixed MLXFP4 model")
args = ap.parse_args()

t0 = time.time()
print(f"[convert] {args.hf_path} -> {args.mlx_path}")
print("[convert] mixed mxfp4(4b)/mxfp8(8b) gs32, dtype bf16, custom predicate")
convert(
    hf_path=args.hf_path,
    mlx_path=args.mlx_path,
    quantize=True,
    q_mode="mxfp4",
    q_bits=4,
    q_group_size=32,
    dtype="bfloat16",
    trust_remote_code=True,
    quant_predicate=pred,
)
print(f"[convert] DONE in {time.time() - t0:.0f}s -> {args.mlx_path}")
