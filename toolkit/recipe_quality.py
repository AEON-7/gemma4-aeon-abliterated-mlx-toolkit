"""
QUALITY-FIRST recipe (revised after empirical finding that MLX mxfp4 E2M1 is lossy):
  mxfp4 (4-bit) : attention q/k/v_proj only  -- the quant-tolerant inputs that do NOT
                  write the residual stream.
  mxfp8 (8-bit) : attention o_proj + MLP gate/up/down  -- the quality-critical MLP bulk
                  AND both abliteration residual-writers (o_proj, down_proj).
  bf16          : tied embed/lm_head + vision/audio projectors + norms.
Goal: match full-mxfp8 fidelity (near-lossless) while remaining a genuine FP4 build and
preserving abliteration + multimodal. ~10.5-11 GB.
"""
SKIP = (
    "embed_tokens", "lm_head", "vision_embedder", "embed_vision", "embed_audio",
    "embedding_post_projection", "per_layer", "altup", "laurel", "layer_scalar", "scaled",
)
FP4 = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")

def pred(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    if any(p in path for p in FP4):
        return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
