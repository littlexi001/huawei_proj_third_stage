#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-8B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/final_hidden_mse_8B_dclm}"

# Build hif8 CUDA once before including hif8:
#   cd Metis/hif8/hif8_cuda && bash build.sh && python hif8_bf16.py && cd ../../..
FORMATS="${FORMATS:-nvfp4,nvfp8,mxfp4,mxfp8,hif4,hif8}"

python eval_final_hidden_mse.py \
  --model_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --formats "${FORMATS}" \
  --max_samples "${MAX_SAMPLES:-32}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --stride "${STRIDE:-2048}" \
  --dtype "${DTYPE:-bfloat16}" \
  --store_dtype "${STORE_DTYPE:-bfloat16}" \
  --q_scalar_w "${Q_SCALAR_W:-1.0}" \
  --q_scalar_x "${Q_SCALAR_X:-1.0}" \
  --device "${DEVICE:-cuda}"
