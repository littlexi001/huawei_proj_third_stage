#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/lym_code/scripts/huawei_proj_third_stage}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/mse_acc_prediction_qwen3_8b}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"

FORMATS="${FORMATS:-hif4,hif8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
PROXY_DEVICE="${PROXY_DEVICE:-cuda}"
SVD_METHOD="${SVD_METHOD:-lowrank}"
SVD_OVERSAMPLE="${SVD_OVERSAMPLE:-8}"
SVD_NITER="${SVD_NITER:-2}"

HIF4_ACC_CSV="${HIF4_ACC_CSV:-${ROOT_DIR}/outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif4/plots/acc_matrix.csv}"
HIF8_ACC_CSV="${HIF8_ACC_CSV:-${ROOT_DIR}/outputs/qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif8_cuda/plots/acc_matrix.csv}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/quant_error-main/quant_error-main/data/eval/mmlu_validation_1531.jsonl}"
RANK_CANDIDATES="${RANK_CANDIDATES:-0,5,10,15,20,25,30,40,50,60,80,100}"
ANCHOR_PAIRS="${ANCHOR_PAIRS:-0:0,0:100,100:0,20:20,40:40,60:60,80:80,100:100,20:80,80:20,40:80,80:40}"

MAX_ACC_DROP="${MAX_ACC_DROP:-0.01}"
LIMIT="${LIMIT:-16}"
MAX_ACT_ROWS="${MAX_ACT_ROWS:-128}"
MAX_MODULES="${MAX_MODULES:-0}"
MODULE_NAME_CONTAINS="${MODULE_NAME_CONTAINS:-}"
RANDOM_ANCHORS="${RANDOM_ANCHORS:-0}"
SEED="${SEED:-0}"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/predict_mse_to_mmlu_acc_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/predict_mse_to_mmlu_acc.pid"

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

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
  --max-act-rows "${MAX_ACT_ROWS}" \
  --max-modules "${MAX_MODULES}" \
  --module-name-contains "${MODULE_NAME_CONTAINS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --proxy-device "${PROXY_DEVICE}" \
  --svd-method "${SVD_METHOD}" \
  --svd-oversample "${SVD_OVERSAMPLE}" \
  --svd-niter "${SVD_NITER}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started predict_mse_to_mmlu_acc.py"
echo "PID: $(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
