# gemma4-aeon-mlx-toolkit

**Apple-Silicon MLX toolkit + OpenAI-compatible server for the Gemma-4-12B AEON Abliterated quant grid.**

[![model: MLXFP4](https://img.shields.io/badge/HF-MLXFP4-blue)](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4)
[![model: MLX-8bit](https://img.shields.io/badge/HF-MLX--8bit-blue)](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit)
[![container](https://img.shields.io/badge/GHCR-gemma4--aeon--mlx--toolkit-purple)](https://github.com/AEON-7/gemma4-aeon-mlx-toolkit/pkgs/container/gemma4-aeon-mlx-toolkit)

This repo + container bundles the **reproducible quantization, validation, and serving pipeline** behind AEON-7's MLX builds of [`Gemma-4-12B-it-AEON-Abliterated-K4-BF16`](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16) — a K=4 multi-direction biprojection abliteration of `google/gemma-4-12B-it` (multimodal `gemma4_unified`).

> **🎯 Intended target hardware: Apple Silicon (M-series).** All benchmarks below are measured on a **MacBook Pro M4 Pro, 48 GB**.

---

## ⚠️ Read first: containers can't touch Metal on macOS

There is **no Metal GPU passthrough in containers on macOS** — Docker runs Linux in a VM with no access to the Apple GPU. So **no container (this one included) can deliver Metal-accelerated MLX inference on a Mac.** For real on-device performance you run the toolkit **host-native** (one `pip install`, below). This image exists as a *versioned, reproducible bundle* of the pipeline + an OpenAI server that runs with Metal **when its scripts are executed on the Mac host**, and on CPU under Linux/cloud.

We also evaluated **vLLM** on Apple Silicon (`vllm-project/vllm-metal`): as of 2026-06, its docs list Gemma-4 as **experimental** and **"multimodal not supported,"** and its text-only auto-path matches `model_type=='gemma4'` exactly — our checkpoint is `gemma4_unified`, so it is **not served** by that path. The dependable, fully-multimodal Apple-Silicon server for this model is **`mlx_vlm.server`**, which is what this toolkit uses.

---

## The quant grid

| Variant | Repo | Precision | Size | top-1 vs BF16 | median KL | Best for |
|---|---|---|---:|---:|---:|---|
| BF16 (source) | [`…-K4-BF16`](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16) | bf16 | 23.9 GB | 1.000 | 0 | Fine-tuning / reference |
| **MLX 8-bit** (near-lossless) | [`…-MLX-8bit`](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit) | mxfp8 + bf16 | 13.4 GB | **0.924** | **0.0019** | Max fidelity, 24 GB+ Macs |
| **MLX FP4** (compact) | [`…-MLXFP4`](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4) | mixed mxfp4/mxfp8 + bf16 | 9.3 GB | 0.885 | 0.0042 | Smallest / fastest, 16 GB Macs |

---

## Quickstart (host-native — Metal accelerated)

```bash
pip install -U mlx-vlm                              # Apple Silicon, Python 3.12 arm64

# OpenAI-compatible server (text + image + audio)
python -m mlx_vlm.server --model AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4 --port 8080

# one-off generation
python -m mlx_vlm.generate \
  --model AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit \
  --image photo.jpg --prompt "Describe this image." --max-tokens 256
```

Call it like any OpenAI endpoint:

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "aeon-gemma4",
  "messages": [{"role":"user","content":"Explain mixed-precision quantization."}]
}'
```

For a production multi-model server, [`FastMLX`](https://github.com/Blaizzy/fastmlx) wraps the same MLX-VLM backend with concurrency + model management.

## Using the container

```bash
docker pull ghcr.io/aeon-7/gemma4-aeon-mlx-toolkit:latest
docker run --rm ghcr.io/aeon-7/gemma4-aeon-mlx-toolkit help
# reproduce a quant / run validation (mounts a working dir):
docker run --rm -v "$PWD:/work" -w /work ghcr.io/aeon-7/gemma4-aeon-mlx-toolkit \
  quantize --hf-path /work/bf16 --mlx-path /work/out
```
*(Inside Docker this runs on CPU — see the Metal note above. Use host-native for speed.)*

---

## Enhancements & refinements

This toolkit is the product of an 8-agent research sweep over MLX quantization best-practices, adversarially verified against the installed `mlx 0.31.2 / mlx-vlm 0.6.1` source and the real Gemma-4 module tree. The load-bearing refinements:

### 1. Mixed-mode FP4 via a custom callable predicate
MLX-VLM's stock `--quant-predicate mixed_*` recipes are **affine-only** and silently ignore `--q-mode mxfp4`. The only route to *true mixed-mode FP4* is a custom callable predicate passed to `mlx_vlm.convert`, returning a per-layer `{group_size, bits, mode}` dict (`recipe.py`). This round-trips cleanly through both `mlx_lm.load` and `mlx_vlm.load`.

### 2. Abliteration-writer protection (the key insight)
The K=4 biprojection edit orthogonalizes **`self_attn.o_proj` and `mlp.down_proj`** — the two residual-stream *writers* — against a 4-D refusal subspace on 24/48 layers. RTN 4-bit noise re-corrupts that subspace, letting refusals return **and** causing a repetition-loop collapse after ~200–300 tokens. We keep both writers at **`mxfp8` (8-bit)**, which empirically preserves the abliteration: **0/100 refusal regression**, no collapse on ≥512-token generations.

### 3. Encoder-free multimodal preservation
Gemma-4 has **no separate ViT/audio tower** — all image/audio fidelity lives in a few small projection linears (`embed_vision`/`embed_audio`/`vision_embedder`). MLX-VLM's `skip_multimodal_module` does **not** match these unified names, so stock convert would 4-bit them. We force them to **bf16**, keeping image + audio inference intact.

### 4. Soft-capped tied head at bf16
Gemma-4 applies `final_logit_softcapping=30` on a tied `embed_tokens`/`lm_head`. We keep it at bf16 — never 4-bit — to protect logit fidelity.

### 5. Empirical finding — MLX `mxfp4` ≠ NVFP4
Unlike NVIDIA's NVFP4 (two-level scaling, near-lossless at 4-bit), MLX's `mxfp4` is **E2M1 — one mantissa bit** — so it measurably diverges from BF16 wherever applied. We measured this directly (table above): only **8-bit `mxfp8`** is genuinely near-lossless on Apple Silicon. Hence the two-tier grid: an 8-bit flagship and a compact FP4. `nvfp4` mode in MLX is **not** recommended (signed-E4M3 block scales, ~137× less dynamic range than NVIDIA, MLX issue #2962).

---

## Validation & benchmarks (MacBook Pro M4 Pro, 48 GB)

| Build | top-1 vs BF16 | median KL (nats) | harmful refusals | ≥512-tok collapse | multimodal | gen tok/s | TTFT | peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLX 8-bit | 0.924 | 0.0019 | 0/8 | none | ✓ image+audio | 16.4 | 314 ms | 13.5 GB |
| MLX FP4 | 0.885 | 0.0042 | 0/8 | none | ✓ image+audio | 21.4 | 301 ms | 10.1 GB |

Methodology: top-1 / KL measured on the BF16 model's own greedy trajectory (the deployment regime); refusal probe on a harmful-instruction set (classified refuse/comply); collapse guard on ≥512-token generations. Scripts: `traj_eval.py`, `validate.py`, `benchmark.py`.

---

## What's in here

```
toolkit/recipe.py            # mixed mxfp4/mxfp8 predicate (compact build)
toolkit/recipe_8bit.py       # all-mxfp8 + bf16 head/projectors (flagship)
toolkit/convert_mixed_fp4.py # the converter
toolkit/preflight.py         # config + predicate dry-run BEFORE converting
toolkit/audit_quant.py       # prove the precision map landed
toolkit/validate.py          # abliteration + coherence gates
toolkit/traj_eval.py         # near-lossless KL / top-1 vs BF16
toolkit/benchmark.py         # tok/s / TTFT / peak RAM
```

## License & responsibility

Inherits the [Gemma license](https://ai.google.dev/gemma/terms). These are **uncensored** models — downstream safety and legal responsibility rest entirely with the operator; see the **Arbitration Clause** on each model card before use. Quantized by AEON-7 on Apple Silicon; recipe designed + adversarially validated with AI-engineering assistance from Anthropic.
