# AGENTS.md — operator guide for AI agents

This file tells an autonomous coding/ops agent how to **set up, serve, and reproduce** the
**Gemma-4-12B-it AEON Abliterated** MLX models on **Apple Silicon**. Read it before acting — it
pre-empts the traps generic docs miss.

## 0. Ground truth (verify, don't assume)
- **Platform: Apple Silicon Macs (M-series) only.** The acceleration is **Metal**.
- Two models — **pick by unified memory**:
  | RAM | Model | Size | Peak RAM | Notes |
  |---|---|---|---|---|
  | **16 GB** | `AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4` | 9.3 GB | ~10 GB | compact, fastest |
  | **24 GB+** | `AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLX-8bit` | 13.4 GB | ~14 GB | near-lossless |
- Architecture is **`gemma4_unified`** (multimodal: text + image + audio). Serve with **mlx-vlm**.

## 1. Hard rules (do NOT violate)
- **Do NOT run inference inside Docker on macOS** — no Metal passthrough → CPU only. Run **host-native**.
- **Do NOT use vllm-metal for these models** — it does not support `gemma4_unified` multimodal; its text-only auto-path matches `model_type=='gemma4'` exactly (ours is `gemma4_unified`). Use `mlx_vlm.server`.
- **Do NOT quantize the abliteration residual-writers (`self_attn.o_proj`, `mlp.down_proj`) below 8-bit** — refusals return and output collapses into repetition after ~200–300 tokens. (Already handled in the shipped builds.)
- Python must be **3.12, arm64** (not Rosetta/x86).

## 2. Setup (host-native)
```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.12 arm64
pip install -U mlx-vlm huggingface_hub
# If the repos are still private, authenticate first:
hf auth login            # or: export HF_TOKEN=hf_xxx   (needs AEON-7 access)
```

## 3. Serve (OpenAI-compatible)
```bash
MODEL=AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4     # or ...-MLX-8bit on 24 GB+
python -m mlx_vlm.server --model "$MODEL" --port 8080
# test — set "temperature":1.0. The server defaults to greedy (temperature 0), which can
# repeat/loop on long prompts; this model is tuned for temperature 1.0 (top_p 0.95, top_k 64):
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"temperature\":1.0}"
```
Toolkit convenience (host-native): `aeon serve` (env `MODEL`, `PORT`).

## 3b. Optional: +MTP speculative decoding (~1.1–1.2×, output-identical)
Google's official Gemma-4 MTP draft (`google/gemma-4-12B-it-assistant`, 423M) proposes tokens the
target verifies — identical output, just faster. **`--draft-block-size 2` is the benchmarked optimum**
on these quants (deeper drafts are *slower* — acceptance decays with depth on abliterated/quantized weights):
```bash
hf download google/gemma-4-12B-it-assistant     # latest official MTP draft (gated; `hf auth login` once)
python -m mlx_vlm.server --model "$MODEL" --port 8080 \
  --draft-model google/gemma-4-12B-it-assistant --draft-kind mtp --draft-block-size 2
# env form (works with `aeon serve`):
#   MLX_VLM_DRAFT_MODEL=google/gemma-4-12B-it-assistant MLX_VLM_DRAFT_KIND=mtp MLX_VLM_DRAFT_BLOCK_SIZE=2 aeon serve
```
Lossless (target verifies every drafted token); +~0.9 GB RAM. Drop the `--draft-*` flags to disable.

## 4. One-off generation (text / image / audio)
```bash
python -m mlx_vlm.generate --model "$MODEL" --prompt "..." --max-tokens 512 --temperature 1.0   # temperature 1.0 = model's native sampling
python -m mlx_vlm.generate --model "$MODEL" --image pic.jpg  --prompt "Describe it." --max-tokens 256
python -m mlx_vlm.generate --model "$MODEL" --audio clip.wav --prompt "Transcribe + summarize." --max-tokens 256
```

## 5. Reproduce the quant (optional)
```bash
python toolkit/preflight.py        --hf-path <bf16-dir>                 # PLE off + 192/136/4 split
python toolkit/convert_mixed_fp4.py --hf-path <bf16-dir> --mlx-path out/mlxfp4   # compact build
python toolkit/audit_quant.py      --mlx-path out/mlxfp4               # prove the precision map landed
```
The near-lossless 8-bit build uses `recipe_8bit.py` (all-`mxfp8` + bf16 head/projectors).

## 6. Validate (gate before publishing anything)
```bash
python toolkit/validate.py  --model out/mlxfp4     # harmful refusals ~0, no >=512-tok collapse
python toolkit/traj_eval.py                        # top-1 / KL vs BF16 (near-lossless check)
python toolkit/benchmark.py --model out/mlxfp4     # tok/s, TTFT, peak RAM
```
**Pass criteria:** harmful refusal rate ≈ 0, no repetition collapse, median KL < ~0.005 nats (typical token).

## 7. Gotchas you WILL hit
- **BOS matters.** Gemma is BOS-sensitive; for raw-text perplexity/KL prepend `<bos>` (id 2). `mlx_vlm.generate`/server apply the chat template for you.
- **mlx_lm cannot load `gemma4_unified`** ("Model type not supported") — use mlx-vlm, or export a `gemma4_text` backbone for mlx_lm-only tools.
- **`mxfp4` ≠ near-lossless** (E2M1, one mantissa bit). The 8-bit build is the near-lossless one.
- **Private repos** need an AEON-7-scoped HF token for `model_info` / pull.

## 8. Uncensored — operator responsibility
Refusals are removed. Implement downstream safety (input/output filtering, logging, human review for
high-risk workflows). See the **Arbitration Clause** on each Hugging Face model card.
