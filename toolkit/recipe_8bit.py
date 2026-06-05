"""
Near-lossless FLAGSHIP recipe: mxfp8 (8-bit) on ALL language-decoder linears,
bf16 on the tied head + vision/audio projectors + norms. The genuine near-lossless
build for Apple Silicon; preserves the vision/audio encoders at full precision.
"""
SKIP = (
    "embed_tokens", "lm_head", "vision_embedder", "embed_vision", "embed_audio",
    "embedding_post_projection", "per_layer", "altup", "laurel", "layer_scalar", "scaled",
)

def pred(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
