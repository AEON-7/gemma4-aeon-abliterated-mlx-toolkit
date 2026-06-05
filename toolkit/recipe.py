"""
AEON-7 · Gemma-4-12B-it-AEON-Abliterated · MLXFP4 mixed-precision quant recipe.

Derived + adversarially validated by the research workflow (run wf_e4553494-907),
grounded against the installed stack (mlx 0.31.2 / mlx-lm 0.31.3 / mlx-vlm 0.6.1).

PRECISION MAP
  mxfp4 (E2M1, 4-bit, gs32) : attention q_proj/k_proj, MLP gate_proj/up_proj
        -> bulk of params, quant-tolerant, read-only INTO the residual stream.
  mxfp8 (E4M3, 8-bit, gs32) : attention o_proj, MLP down_proj, attention v_proj
        -> residual-WRITERS that carry the K4 multi-direction-biprojection
           abliteration edit. 4-bit noise here re-injects energy along the
           ablated refusal subspace (refusals return) and causes the documented
           "repetition-loop collapse after ~200-300 tokens". Keep >= 8-bit.
  bf16 (explicit skip)      : tied embed_tokens/lm_head (final_logit_softcapping=30,
           precision-sensitive) and the encoder-free modality projectors
           (vision_embedder / embed_vision / embed_audio) -- all image+audio
           fidelity lives in these few small Linears, and skip_multimodal_module()
           does NOT catch their unified names (verified False on mlx-vlm 0.6.1).
  bf16 (automatic)          : all RMSNorm / QK-norm / value-norm / layer_scalar
           (1-D tensors -> never reach the quantizer).

Expected quantizable-Linear counts on the 12B: 192 mxfp4 / 136 mxfp8 / 4 bf16-skip.

IMPORTANT: when a custom callable is passed to mlx_vlm.convert(), it REPLACES the
default base predicate, so the SKIP list below must itself cover the projectors.
"""

# Substrings -> keep bf16 (do NOT quantize).
SKIP = (
    "embed_tokens", "lm_head",                 # tied, soft-capped head
    "vision_embedder", "embed_vision", "embed_audio",  # modality projectors (encoder-free)
    "embedding_post_projection",
    "per_layer", "altup", "laurel",            # PLE/AltUp guards (off on 12B; harmless if absent)
    "layer_scalar", "scaled",
)

# Substrings -> mxfp8 (8-bit). Residual-writers + sliding-layer v_proj.
PROTECT_8 = ("self_attn.o_proj", "mlp.down_proj", "self_attn.v_proj")

MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def pred(path, module):
    """Per-layer class predicate for mlx_lm.quantize_model.

    Returns False (skip -> bf16), or a dict {group_size, bits, mode}.
    """
    # Only quantizable layers expose to_quantized(); norms/scalars never do.
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    if any(p in path for p in PROTECT_8):
        return dict(MXFP8)
    return dict(MXFP4)
