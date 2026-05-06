#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="../../models/Qwen3-0.6B"
OUTPUT_DIR="./outputs/qwen3_svd_resid_nvfp4_mmlu_grid_fixed"
SCRIPT="qwen3_svd_residual_nvfp4_mmlu.py"

# 可选：如果 Metis 不在当前环境路径里，取消下一行注释并改成你的项目路径
# export PYTHONPATH=/mnt/workspace/your_project:${PYTHONPATH}
# source /path/to/venv/bin/activate

mkdir -p "${OUTPUT_DIR}"

nohup python "${SCRIPT}" \
  --mode grid \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpu_ids "0,1,2,3,4,5,6,7" \
  --rank_grid "0,5,10,15,20,25,30,40,50,60,80,100" \
  --run_unquantized_baseline \
  --qtype "nvfp4e2m1bnosr" \
  --q_scalar_w 1.0 \
  --q_scalar_x 1.0 \
  --svd_method_w randomized \
  --svd_method_x randomized \
  --dtype bfloat16 \
  --store_dtype bfloat16 \
  --max_length 2048 \
  --ntrain 0 \
  --split validation \
  --max_eval_samples_per_subject -1 \
  > "${OUTPUT_DIR}/launcher.log" 2>&1 &

echo "launched. log: ${OUTPUT_DIR}/launcher.log"
echo "status: ${OUTPUT_DIR}/grid_status.jsonl"
