#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-./outputs/qwen3_svd_resid_hif4_gsm8k_grid_8B}"

python plot_rank_heatmaps.py \
  --root "${ROOT}" \
  --task_name GSM8K
