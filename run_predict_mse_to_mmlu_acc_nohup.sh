#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Predict MMLU accuracy from output-level hidden-state MSE.
#
# For every (ka, kb) grid point, this script:
#   1. SVD-decomposes all Linear weights at rank=kb, quantizes main & residual
#   2. Runs forward pass with activation SVD-decomposition at rank=ka
#   3. Computes last-hidden-state MSE vs uncompressed baseline
#   4. Fits ridge regression: output_mse → MMLU accuracy drop
#   5. Predicts accuracy for all grid points, outputs feasible configurations
#
# NOTE: First run is SLOW (144 model loads + forward passes per format).
#       Results are cached to proxy_mse_cache.json for subsequent fast runs.
# ============================================================================

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/lym_code/scripts/huawei_proj_third_stage}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/mse_acc_prediction_qwen3_8b}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"

FORMATS="${FORMATS:-hif4,hif8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SVD_METHOD="${SVD_METHOD:-randomized}"

HIF4_ACC_CSV="${HIF4_ACC_CSV:-${ROOT_DIR}/outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif4/plots/acc_matrix.csv}"
HIF8_ACC_CSV="${HIF8_ACC_CSV:-${ROOT_DIR}/outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif8_cuda/plots/acc_matrix.csv}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/quant_error-main/quant_error-main/data/eval/mmlu_validation_1531.jsonl}"
RANK_CANDIDATES="${RANK_CANDIDATES:-0,5,10,15,20,25,30,40,50,60,80,100}"
ANCHOR_PAIRS="${ANCHOR_PAIRS:-0:0,0:100,100:0,20:20,40:40,60:60,80:80,100:100,20:80,80:20,40:80,80:40}"

MAX_ACC_DROP="${MAX_ACC_DROP:-0.01}"
LIMIT="${LIMIT:-20}"
MODULE_NAME_CONTAINS="${MODULE_NAME_CONTAINS:-}"
RANDOM_ANCHORS="${RANDOM_ANCHORS:-0}"
SEED="${SEED:-0}"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/predict_mse_to_mmlu_acc_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/predict_mse_to_mmlu_acc.pid"

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup "${PYTHON_BIN}" "${ROOT_DIR}/predict_mse_to_mmlu_acc.py" \
  --model-path "${MODEL_PATH}" \
  --formats "${FORMATS}" \
  --hif4-acc-csv "${HIF4_ACC_CSV}" \
  --hif8-acc-csv "${HIF8_ACC_CSV}" \
  --data "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --rank-candidates "${RANK_CANDIDATES}" \
  --anchor-pairs "${ANCHOR_PAIRS}" \
  --random-anchors "${RANDOM_ANCHORS}" \
  --seed "${SEED}" \
  --max-acc-drop "${MAX_ACC_DROP}" \
  --limit "${LIMIT}" \
  --module-name-contains "${MODULE_NAME_CONTAINS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --svd-method "${SVD_METHOD}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started predict_mse_to_mmlu_acc.py (output MSE mode)"
echo "PID: $(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "WARNING: First run will load model 144+ times per format and run forward passes."
echo "         Subsequent runs will use cached proxy_mse_cache.json (instant)."
