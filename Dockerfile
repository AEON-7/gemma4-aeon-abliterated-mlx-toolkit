# Gemma-4-12B-it AEON Abliterated — Apple Silicon MLX toolkit
#
# IMPORTANT: Apple's Metal GPU is NOT accessible inside containers on macOS (no GPU
# passthrough in Docker's Linux VM). For Metal-accelerated inference, run the bundled
# scripts HOST-NATIVE on your Mac (see README "Quickstart, host-native"). This image is
# a versioned, reproducible bundle of the quant + validation + serve pipeline and an
# OpenAI-compatible server (Metal when run on the Mac host; CPU under Linux/cloud).
FROM python:3.12-slim

LABEL org.opencontainers.image.title="gemma4-aeon-mlx-toolkit"
LABEL org.opencontainers.image.description="Apple Silicon MLX toolkit + OpenAI-compatible server for the Gemma-4-12B AEON Abliterated MLX quant grid (MLXFP4 / MLX-8bit)"
LABEL org.opencontainers.image.source="https://github.com/AEON-7/gemma4-aeon-mlx-toolkit"
LABEL org.opencontainers.image.licenses="Gemma"
LABEL org.opencontainers.image.vendor="AEON-7"

ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL=AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4 \
    PORT=8080

WORKDIR /opt/toolkit

# huggingface_hub always installs; mlx-vlm is installed lazily by the entrypoint so the
# image builds on any platform (MLX wheels are Apple-Silicon-first).
RUN pip install --upgrade pip && pip install "huggingface_hub[cli]"

COPY toolkit/ /opt/toolkit/
COPY entrypoint.sh /usr/local/bin/aeon
RUN chmod +x /usr/local/bin/aeon

ENTRYPOINT ["/usr/local/bin/aeon"]
CMD ["help"]
