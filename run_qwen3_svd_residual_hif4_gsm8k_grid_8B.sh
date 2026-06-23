#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3_svd_resid_hif4_gsm8k_grid_8B}"
SCRIPT="${SCRIPT:-qwen3_svd_residual_nvfp4_gsm8k.py}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RANK_GRID="${RANK_GRID:-0,5,10,15,20,25,30,40,50,60,80,100}"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" --mode download_data

nohup python "${SCRIPT}" \
  --mode grid \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --rank_grid "${RANK_GRID}" \
  --run_unquantized_baseline \
  --qtype "hif4_0418" \
  --blocksize 16 \
  --q_scalar_w 1.0 \
  --q_scalar_x 1.0 \
  --svd_method_w randomized \
  --svd_method_x randomized \
  --dtype bfloat16 \
  --store_dtype bfloat16 \
  --max_length 2048 \
  --max_new_tokens 256 \
  --ntrain 8 \
  --split test \
  --max_eval_samples -1 \
  > "${OUTPUT_DIR}/launcher.log" 2>&1 &

echo "launched. log: ${OUTPUT_DIR}/launcher.log"
echo "status: ${OUTPUT_DIR}/grid_status.jsonl"
