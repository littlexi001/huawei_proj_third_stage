#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace/lym_code/scripts/huawei_proj_third_stage/outputs/qwen3_svd_resid_nvfp4_mmlu_grid_fixed"
BASELINE="0.4022"

python plot_rank_heatmaps.py \
  --root "${ROOT}" \
  --baseline "${BASELINE}"
