#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the third-stage Adaptive-Metis experiment.
#
# Required:
#   MODEL_PATH=/path/to/hf/model bash run_adaptive_metis_fp8_experiment.sh
#
# Common overrides:
#   FORMATS="hif8 fp8 nvfp8"
#   DEVICE=cuda
#   DTYPE=auto
#   RUN_VERIFY=1

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: set MODEL_PATH first, for example:"
  echo "  MODEL_PATH=/path/to/Qwen3-0.6B bash run_adaptive_metis_fp8_experiment.sh"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')}"

FORMATS="${FORMATS:-hif8 fp8 nvfp8}"
RANK_CANDIDATES="${RANK_CANDIDATES:-0,5,10,15,20,25,30,40,50,60,80,100}"
ANCHOR_PAIRS="${ANCHOR_PAIRS:-0:0,0:100,100:0,20:20,40:40,60:60,80:80,100:100,20:80,80:20,40:80,80:40}"

PROXY_DATA="${PROXY_DATA:-data/eval/mmlu_validation_1531.jsonl}"
ORACLE_DATA="${ORACLE_DATA:-data/eval/mmlu_test_14042.jsonl}"
OUT_DIR="${OUT_DIR:-data/results/adaptive_metis_fp8}"

PROXY_LIMIT="${PROXY_LIMIT:-256}"
MAX_ACT_ROWS="${MAX_ACT_ROWS:-512}"
MAX_MODULES="${MAX_MODULES:-0}"
MODULE_NAME_CONTAINS="${MODULE_NAME_CONTAINS:-}"

CALIBRATION_SIZE="${CALIBRATION_SIZE:-256}"
ANCHOR_EVAL_LIMIT="${ANCHOR_EVAL_LIMIT:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SVD_DEVICE="${SVD_DEVICE:-auto}"

MAX_ACCURACY_DROP="${MAX_ACCURACY_DROP:-0.01}"
UNCERTAINTY_LAMBDA="${UNCERTAINTY_LAMBDA:-1.0}"
KA_COST="${KA_COST:-1.0}"
KP_COST="${KP_COST:-1.0}"

RUN_VERIFY="${RUN_VERIFY:-0}"
VERIFY_EVAL_LIMIT="${VERIFY_EVAL_LIMIT:-14042}"

mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Formats: ${FORMATS}"
echo "Output directory: ${OUT_DIR}"

for FORMAT in ${FORMATS}; do
  PREFIX="${OUT_DIR}/${MODEL_TAG}_${FORMAT}"
  PROXY_OUT="${PREFIX}_rank_search.json"
  ANCHOR_OUT="${PREFIX}_anchors.jsonl"
  SELECT_OUT="${PREFIX}_selected.json"
  VERIFY_OUT="${PREFIX}_verify_best.jsonl"

  echo
  echo "=== [${FORMAT}] 1/3 dense proxy MSE sweep ==="
  "${PYTHON_BIN}" scripts/adaptive_metis_rank_search.py \
    --model-path "${MODEL_PATH}" \
    --data "${PROXY_DATA}" \
    --task mmlu \
    --format "${FORMAT}" \
    --rank-candidates "${RANK_CANDIDATES}" \
    --limit "${PROXY_LIMIT}" \
    --max-act-rows "${MAX_ACT_ROWS}" \
    --max-modules "${MAX_MODULES}" \
    --module-name-contains "${MODULE_NAME_CONTAINS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --output "${PROXY_OUT}"

  echo
  echo "=== [${FORMAT}] 2/3 sparse MMLU anchor evaluation ==="
  "${PYTHON_BIN}" scripts/adaptive_metis_oracle_sweep.py \
    --model-path "${MODEL_PATH}" \
    --data "${ORACLE_DATA}" \
    --format "${FORMAT}" \
    --rank-candidates "${RANK_CANDIDATES}" \
    --pairs "${ANCHOR_PAIRS}" \
    --calibration-size "${CALIBRATION_SIZE}" \
    --eval-limit "${ANCHOR_EVAL_LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --svd-device "${SVD_DEVICE}" \
    --ka-cost "${KA_COST}" \
    --kp-cost "${KP_COST}" \
    --max-modules "${MAX_MODULES}" \
    --output "${ANCHOR_OUT}"

  echo
  echo "=== [${FORMAT}] 3/3 fit anchors and select feasible configs ==="
  "${PYTHON_BIN}" scripts/select_metis_configs.py \
    --proxy "${PROXY_OUT}" \
    --anchors "${ANCHOR_OUT}" \
    --max-accuracy-drop "${MAX_ACCURACY_DROP}" \
    --uncertainty-lambda "${UNCERTAINTY_LAMBDA}" \
    --ka-cost "${KA_COST}" \
    --kp-cost "${KP_COST}" \
    --output "${SELECT_OUT}"

  echo "Selected result: ${SELECT_OUT}"

  if [[ "${RUN_VERIFY}" == "1" ]]; then
    BEST_PAIR="$("${PYTHON_BIN}" - "${SELECT_OUT}" <<'PY'
import json
import sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
best = data.get("best_min_cost_feasible")
if not best:
    raise SystemExit("No feasible config found; skip verify.")
print(f'{int(best["ka"])}:{int(best["kp"])}')
PY
)"
    echo
    echo "=== [${FORMAT}] verify best config ${BEST_PAIR} ==="
    "${PYTHON_BIN}" scripts/adaptive_metis_oracle_sweep.py \
      --model-path "${MODEL_PATH}" \
      --data "${ORACLE_DATA}" \
      --format "${FORMAT}" \
      --rank-candidates "${RANK_CANDIDATES}" \
      --pairs "${BEST_PAIR}" \
      --calibration-size "${CALIBRATION_SIZE}" \
      --eval-limit "${VERIFY_EVAL_LIMIT}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --svd-device "${SVD_DEVICE}" \
      --ka-cost "${KA_COST}" \
      --kp-cost "${KP_COST}" \
      --max-modules "${MAX_MODULES}" \
      --output "${VERIFY_OUT}"
    echo "Verify result: ${VERIFY_OUT}"
  fi
done

echo
echo "Done. Main outputs are under ${OUT_DIR}"
