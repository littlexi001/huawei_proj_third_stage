#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:-./outputs/final_hidden_mse_8B_dclm/summary.csv}"
METRICS="${METRICS:-mse,relative_mse}"

python plot_final_hidden_mse_heatmaps.py \
  --csv "${CSV_PATH}" \
  --metrics "${METRICS}" \
  --dpi "${DPI:-220}" \
  ${ANNOTATE:+--annotate} \
  ${LOG_SCALE:+--log}
