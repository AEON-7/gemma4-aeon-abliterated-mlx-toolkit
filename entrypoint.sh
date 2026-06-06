#!/usr/bin/env bash
# AEON-7 Gemma-4 MLX toolkit entrypoint.
set -euo pipefail

ensure_mlx() {
  python -c "import mlx_vlm" 2>/dev/null || {
    echo "[aeon] installing mlx-vlm (Apple-Silicon-first; CPU under Linux)..." >&2
    pip install -q mlx mlx-vlm || {
      echo "[aeon] ERROR: MLX could not be installed on this platform." >&2
      echo "[aeon] Metal acceleration requires running host-native on Apple Silicon." >&2
      echo "[aeon] See: https://github.com/AEON-7/gemma4-aeon-abliterated-mlx-toolkit#quickstart" >&2
      exit 2
    }
  }
}

MODEL="${MODEL:-AEON-7/Gemma-4-12B-it-AEON-Abliterated-MLXFP4}"

case "${1:-help}" in
  serve)
    ensure_mlx; shift
    echo "[aeon] mlx_vlm.server  model=$MODEL port=${PORT:-8080}" >&2
    exec python -m mlx_vlm.server --model "$MODEL" --port "${PORT:-8080}" "$@" ;;
  generate)
    ensure_mlx; shift
    exec python -m mlx_vlm.generate --model "$MODEL" "$@" ;;
  quantize)
    ensure_mlx; shift
    exec python /opt/toolkit/convert_mixed_fp4.py "$@" ;;
  validate)
    ensure_mlx; shift
    exec python /opt/toolkit/validate.py "$@" ;;
  benchmark)
    ensure_mlx; shift
    exec python /opt/toolkit/benchmark.py "$@" ;;
  version)
    python -c "import importlib.metadata as m;[print(p, m.version(p)) for p in ['mlx','mlx-vlm','huggingface-hub']]" 2>/dev/null || echo "mlx not yet installed (run any command to install)";;
  help|--help|-h|*)
    cat /opt/toolkit/USAGE.txt ;;
esac
