#!/usr/bin/env bash
set -euo pipefail

# Override ROOT when plotting another experiment, for example:
#   ROOT=./outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_mxfp8 bash run_plot_rank_heatmaps.sh
ROOT="${ROOT:-./outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_nvfp8}"

if [[ ! -d "${ROOT}" ]]; then
  echo "[ERROR] root not found: ${ROOT}" >&2
  echo "[INFO] available output dirs:" >&2
  find ./outputs -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort >&2 || true
  exit 1
fi

echo "[INFO] processing ${ROOT}"
python plot_rank_heatmaps.py --root "${ROOT}"
