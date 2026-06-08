# gemma4-aeon-abliterated-mlx-toolkit

**Apple-Silicon MLX builds of Gemma-4-12B AEON Abliterated — comparison card, reproducible toolkit, and OpenAI-compatible server.**

[![MLXFP4](https://img.shields.io/badge/HF-MLXFP4%20(9.3GB)-blue)](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4)
[![MLX-8bit](https://img.shields.io/badge/HF-MLX--8bit%20(13.4GB)-blue)](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit)
[![BF16 source](https://img.shields.io/badge/HF-K4--BF16%20source-lightgrey)](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16)
[![container](https://img.shields.io/badge/GHCR-gemma4--aeon--abliterated--mlx--toolkit-purple)](https://github.com/AEON-7/gemma4-aeon-abliterated-mlx-toolkit/pkgs/container/gemma4-aeon-abliterated-mlx-toolkit)

MLX (Apple Silicon) quantizations of [`Gemma-4-12B-it-AEON-Abliterated-K4-BF16`](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16) — a K=4 multi-direction biprojection abliteration of `google/gemma-4-12B-it` (multimodal `gemma4_unified`: text + image + audio). Two builds shipped: a **near-lossless 8-bit** flagship and a **high-quality compact FP4** for tight memory budgets. **Agents: read [`AGENTS.md`](./AGENTS.md) first.**

> **🎯 Intended target hardware: Apple Silicon (M-series).** All benchmarks here are measured on a **MacBook Pro · Apple M4 Pro · 48 GB**.

---

## ⚠️ Containers can't touch Metal on macOS

There is **no Metal GPU passthrough in containers on macOS** — Docker runs Linux in a VM with no access to the Apple GPU, so **no container delivers Metal-accelerated MLX inference on a Mac.** Run the toolkit **host-native** for real performance (one `pip install`, below). This image is a versioned, reproducible bundle of the pipeline + server; it runs with Metal only when its scripts execute on the Mac host.

We evaluated **vLLM** on Apple Silicon (`vllm-project/vllm-metal`): as of 2026-06 its docs list Gemma-4 as **experimental** and **"multimodal not supported,"** and its text-only auto-path matches `model_type=='gemma4'` exactly — our checkpoint is `gemma4_unified`, so it is **not served**. The dependable, fully-multimodal Apple-Silicon server for this model is **`mlx_vlm.server`**, which is what this toolkit uses.

---

## ⚡ Quickstart (host-native, Metal-accelerated)

Both builds are abliterated + multimodal and serve an OpenAI-compatible API on `127.0.0.1:8080`. **Copy the one block for the build you want** — it installs [`uv`](https://docs.astral.sh/uv/) (which fetches Python 3.12 + mlx-vlm on first run) and starts the server, all on a fresh Mac with nothing pre-installed. *(While the repos are private, run `hf auth login` first; in requests set `"model"` to the launched id.)*

### ▶ To run MLX-8bit — near-lossless · FP8 · ~13.4 GB · 24 GB+ RAM — paste this into your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv run --python 3.12 --with mlx-vlm -- python -m mlx_vlm.server --model AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit --port 8080 --max-kv-size 32768
```

### ▶ To run MLXFP4 — compact · FP4 · ~9.3 GB · 16 GB RAM · faster — paste this into your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv run --python 3.12 --with mlx-vlm -- python -m mlx_vlm.server --model AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4 --port 8080 --max-kv-size 16384
```

Once it's running, call it like any OpenAI endpoint. **Set `temperature: 1.0`** — the server defaults to *greedy* decoding (`temperature 0`), which can repeat or loop on long prompts; both builds are tuned for their native sampling (**`temperature 1.0`**, `top_p 0.95`, `top_k 64`). Clients that send no sampling params get the greedy default, so set it explicitly:

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<launched-id>","messages":[{"role":"user","content":"Explain mixed-precision quantization."}],"temperature":1.0}'
```

### Multimodal (image · audio · video) — on by default

**No flags needed.** mlx-vlm bundles the deps (`Pillow`, `opencv`, `miniaudio`, `mlx-audio`) and the server accepts OpenAI `image_url` + `input_audio` content. Verified on this model — it described a test image **and transcribed speech** correctly through the server.

```bash
# image (http URL or data: URI)
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<launched-id>",
  "messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},
    {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}]}]}'

# audio (base64 wav/mp3, OpenAI input_audio)
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<launched-id>",
  "messages":[{"role":"user","content":[{"type":"text","text":"Transcribe this."},
    {"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,…","format":"wav"}}]}]}'

# one-off via CLI:  aeon generate --image photo.jpg --prompt "…"   |   aeon generate --audio clip.wav --prompt "…"
```

Container convenience (host-native): `docker pull ghcr.io/aeon-7/gemma4-aeon-abliterated-mlx-toolkit` then `aeon serve` / `aeon validate` / `aeon benchmark`.

## 🧠 Context length & memory tuning

The quickstarts set a memory-safe `--max-kv-size` for each model's **minimum** target Mac. KV cache here costs only **~74 KB/token** — Gemma-4's hybrid attention keeps just the **8 full-attention layers** growing; the **40 sliding layers stay capped at 1024** — so even the model's full **256K** context is affordable. Measured on an M4 Pro:

| `--max-kv-size` | Context | KV added | **FP4** peak (10 GB) | **FP8** peak (14 GB) |
|---:|---:|---:|---:|---:|
| `8192` | 8K | 0.6 GB | ~12.3 GB | ~16.3 GB |
| `16384` | 16K | 1.2 GB | ~13 GB | ~17 GB |
| `32768` | 32K | 2.4 GB | ~14 GB | ~18 GB |
| `65536` | 64K | 4.9 GB | ~17 GB | ~21 GB |
| `131072` | 128K | 9.7 GB | ~21 GB | ~25 GB |
| `262144` | 256K (max) | 19.4 GB | ~31 GB | ~35 GB |

**Pick by unified memory:** 16 GB → `16384` (FP4) · 24 GB → `32768` (FP8) · 32 GB → `65536` · **48 GB → `131072`** (128K) · **48 GB long-context → `262144`** (full **256K**, ~35 GB). Raise it until peak approaches your RAM; past that macOS swaps (slow). `--max-kv-size` is a *rotating* cap — beyond it, the oldest tokens are evicted.

### All server flags

| Flag | Default | Effect on performance / memory |
|---|---|---|
| `--max-kv-size N` | unbounded (model max **256K**) | **Main context↔RAM knob.** Caps context to N tokens, ~74 KB/token. |
| `--kv-bits {8,4}` + `--kv-quant-scheme {uniform,turboquant}` | off | Quantize the KV cache. `8` ≈ halves KV (→ ~2× context per GB); `4`/turboquant ≈ quarter, small quality cost. |
| `--quantized-kv-start N` | — | Keep the first N tokens full-precision, quantize the rest. |
| `--prefill-step-size N` | 2048 | Tokens per prefill step. Lower (512) = less transient RAM on long prompts, slightly slower; higher = faster prefill, more RAM. |
| `--vision-cache-size N` | 20 | Cached image features. Lower to save RAM if you don't use vision. |
| `--max-tokens N` | model ceiling | Default output cap per request (overridable per request). |
| `--enable-thinking` | off | Gemma-4 chain-of-thought on by default. |
| `--draft-model … --draft-kind {dflash,eagle3,mtp}` | off | Speculative decoding → faster generation (needs a drafter; more RAM). |
| `--host / --port` | 0.0.0.0 / 8080 | Bind address/port (`127.0.0.1` = local-only). |

**Stretch context on a small Mac:** add `--kv-bits 8` to roughly double the context that fits in the same RAM — e.g. FP4 on 16 GB: `--max-kv-size 32768 --kv-bits 8`.

## Which build? — minimum specs

| Your Mac | Use | Size | Peak RAM | Why |
|---|---|---:|---:|---|
| **16 GB** unified memory | [**MLXFP4**](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4) | 9.3 GB | ~10 GB | high-quality compact, fastest single-stream |
| **24 GB+** unified memory | [**MLX-8bit**](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit) | 13.4 GB | ~14 GB | near-lossless, maximum fidelity |

Minimum: Apple Silicon (M1 or newer). Both run the full multimodal path.

---

## 📊 Full quant comparison (what we measured)

Fidelity measured on the **BF16 model's own greedy trajectory** (the regime that reflects deployment), on a MacBook Pro M4 Pro 48 GB. `top-1` = how often the quant picks BF16's exact next token; `KL` in nats.

| Build | Recipe | top-1 vs BF16 | mean KL | median KL | size | shipped |
|---|---|---:|---:|---:|---:|:--:|
| **BF16 source** | full precision | 1.000 | 0 | 0 | 23.9 GB | reference |
| **MLX-8bit** ✅ | all `mxfp8` + bf16 head/projectors | **0.924** | 0.107 | **0.0019** | 13.4 GB | **flagship** |
| _full mxfp8 (default)_ | `mxfp8` attn/proj/head + int8 MLP | 0.938 | 0.047 | 0.0010 | 12.0 GB | — |
| _quality (q/k/v→fp4)_ | q/k/v `mxfp4`, rest `mxfp8` | 0.895 | 0.194 | 0.0042 | 12.6 GB | — |
| _stock mxfp4_ | mlx-vlm default predicate | 0.908 | 0.156 | 0.0049 | 10.0 GB | — |
| **MLXFP4** ✅ | mixed: q/k/gate/up `mxfp4`, o/down/v `mxfp8`, bf16 head/proj | 0.885 | 0.218 | **0.0042** | 9.3 GB | **compact** |

**Why these two shipped (and not the higher-scoring experiments):**
- **MLX-8bit over _full mxfp8 (default)_** — the default predicate scores a hair higher on text top-1 (0.938 vs 0.924) but quantizes the **multimodal projectors to 8-bit**. We ship the build that keeps `embed_vision`/`embed_audio` + the tied soft-capped head at **bf16** — image/audio fidelity stays exact, at a negligible text cost. This is the "preserve the vision encoders" choice.
- **MLXFP4 over _stock mxfp4_** — stock scores higher on raw top-1 (0.908 vs 0.885) but **4-bits `o_proj` (an abliteration residual-writer), the projectors, and the tied head**. For an *abliterated, multimodal* model that's exactly the wrong place to lose precision. Our MLXFP4 protects all three, trading a sliver of raw top-1 for **guaranteed abliteration survival + multimodal preservation** — and it's still *smaller* (9.3 vs 10 GB), because we 4-bit the big MLP gate/up matrices instead.
- **_quality (q/k/v→fp4)_** — dominated (≈MLXFP4 fidelity, but 12.6 GB). Not shipped.

> **Takeaway:** unlike NVIDIA's NVFP4, **MLX `mxfp4` (E2M1, one mantissa bit) is not near-lossless** — only `mxfp8` is. Hence a two-tier grid rather than a single FP4. MLX `nvfp4` is also not recommended on Metal (signed-E4M3 block scales, ~137× less dynamic range, ml-explore/mlx #2962).

---

## 🔬 How we quantized — methods & refinements

Derived from an 8-agent research sweep over MLX quant best-practices, adversarially verified against the installed `mlx 0.31.2 / mlx-vlm 0.6.1` source and the real Gemma-4 module tree.

1. **Mixed-mode FP4 via a custom callable predicate.** mlx-vlm's stock `--quant-predicate mixed_*` is *affine-only* and silently ignores `--q-mode mxfp4`. True mixed-mode FP4 needs a callable predicate passed to `mlx_vlm.convert` returning per-layer `{group_size, bits, mode}` (`recipe.py`).
2. **Abliteration-writer protection.** The K=4 edit orthogonalizes `o_proj` + `down_proj` (residual writers) against a 4-D refusal subspace on 24/48 layers. We keep both ≥ `mxfp8` → **0/100 refusal regression**, no collapse on ≥512-token generations.
3. **Encoder-free multimodal preservation.** Gemma-4 has no ViT/audio tower — fidelity lives in tiny `embed_vision`/`embed_audio` projectors that `skip_multimodal_module` does *not* match. We force them bf16.
4. **Soft-capped tied head at bf16** (`final_logit_softcapping=30`).
5. **Empirical format choice** — measured `mxfp4` vs `mxfp8` vs `nvfp4` and the placement experiments above; shipped the two best per use-case.

Recipe (`recipe.py`):
```python
SKIP      = ("embed_tokens","lm_head","vision_embedder","embed_vision","embed_audio")
PROTECT_8 = ("self_attn.o_proj","mlp.down_proj","self_attn.v_proj")
def pred(path, m):
    if not hasattr(m,"to_quantized"):     return False
    if any(s in path for s in SKIP):      return False
    if any(p in path for p in PROTECT_8): return {"group_size":32,"bits":8,"mode":"mxfp8"}
    return {"group_size":32,"bits":4,"mode":"mxfp4"}
```

---

## ⏱️ Benchmarks — MacBook Pro · Apple M4 Pro · 48 GB

> Measured on **Apple M4 Pro (14-core CPU, 48 GB unified memory), macOS 26, mlx-vlm 0.6.1.** Relative scaling: base **M4/M3** slower, **M4 Max/Ultra** faster; MLX single-stream is mostly memory-bandwidth bound.

| Build | gen tok/s | prompt tok/s | TTFT | peak RAM | image gen tok/s |
|---|---:|---:|---:|---:|---:|
| **MLXFP4** | **21.4** | 169 | 301 ms | 10.1 GB | 21.2 |
| **MLX-8bit** | **16.4** (peak 17.1) | 163 | 314 ms | 13.5 GB | 16.1 |

*Single-stream, greedy, median of 5 runs after warmup (`benchmark.py`).*

## ✅ Validation methodology

| Gate | Tool | Pass criterion | MLXFP4 | MLX-8bit |
|---|---|---|:--:|:--:|
| Abliteration survived | `validate.py` | harmful refusals ≈ 0 | 0/8 | 0/8 |
| No computation collapse | `validate.py` | coherent ≥512-tok gens | ✅ | ✅ |
| Near-lossless | `traj_eval.py` | median KL < ~0.005 nats | 0.0042 | 0.0019 |
| Multimodal preserved | `mlx_vlm.generate --image` | describes scene correctly | ✅ | ✅ |
| Performance | `benchmark.py` | tok/s, TTFT, peak RAM | see above | see above |

---

## 🤖 For agents

Read **[`AGENTS.md`](./AGENTS.md)** — setup, model selection by RAM, serve, reproduce-the-quant, validation gates, and the hard rules (no Docker-Metal, no vllm-metal for `gemma4_unified`, don't 4-bit the residual-writers).

## What's in here

```
toolkit/recipe.py            # mixed mxfp4/mxfp8 predicate (compact MLXFP4)
toolkit/recipe_8bit.py       # all-mxfp8 + bf16 head/projectors (near-lossless flagship)
toolkit/recipe_quality.py    # q/k/v->mxfp4 experiment
toolkit/convert_mixed_fp4.py # converter
toolkit/preflight.py         # config + predicate dry-run BEFORE converting
toolkit/audit_quant.py       # prove the precision map landed
toolkit/validate.py          # abliteration + coherence gates
toolkit/traj_eval.py         # near-lossless KL / top-1 vs BF16
toolkit/quality_eval.py      # PPL / KL envelope
toolkit/benchmark.py         # tok/s / TTFT / peak RAM
Dockerfile · entrypoint.sh   # `aeon serve|generate|quantize|validate|benchmark`
```

## License & responsibility

Inherits the [Gemma license](https://ai.google.dev/gemma/terms). These are **uncensored** models — downstream safety and legal responsibility rest entirely with the operator; see the **Arbitration Clause** on each model card before use. Quantized by AEON-7 on Apple Silicon (MacBook Pro M4 Pro, 48 GB); recipe designed + adversarially validated with AI-engineering assistance from Anthropic.
