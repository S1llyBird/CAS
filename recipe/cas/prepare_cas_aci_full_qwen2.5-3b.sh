#!/usr/bin/env bash
set -euo pipefail
set -x

ulimit -n 65535

# Use GPUs 0 and 1 by default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data"

# Input data (the original RL training set).
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/cas_train.parquet}"

# Output directory for calibration_dataset, rl_train_dataset, lambda_fixed, and initial_scores.
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/calibration}"

# Retrieval service configuration.
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-20}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-30}"

# ACI base model used to generate initial_scores.pt.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# Calibration hyperparameters; override them as needed.
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-150}"
CP_ALPHA="${CP_ALPHA:-0.20}"
CP_K_MAX="${CP_K_MAX:-10}"
APS_TEMPERATURE="${APS_TEMPERATURE:-0.01}"
APS_ALPHA="${APS_ALPHA:-0.10}"
APS_MIN_DOCS="${APS_MIN_DOCS:-2}"
APS_MAX_DOCS="${APS_MAX_DOCS:-5}"
SEED="${SEED:-42}"

# Use a teacher LLM to recover missing Hotpot hops; enabled by default.
LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
LLM_MODEL="${LLM_MODEL:-deepseek-chat}"
LLM_TIMEOUT="${LLM_TIMEOUT:-60}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_API_KEY_ENV="${LLM_API_KEY_ENV:-OPENAI_API_KEY}"

# Validate the API key before starting the pipeline.
if [[ -z "${!LLM_API_KEY_ENV:-}" ]]; then
  echo "[ERROR] Environment variable ${LLM_API_KEY_ENV} is not set; --enable_llm_hop_extraction cannot run."
  echo "Set it first: export ${LLM_API_KEY_ENV}=<your_api_key>"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python3 "${PROJECT_DIR}/scripts/prepare_calibration.py" \
  --train_parquet "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --calib_sample_size "${CALIB_SAMPLE_SIZE}" \
  --seed "${SEED}" \
  --retrieval_service_url "${RETRIEVAL_SERVICE_URL}" \
  --retrieval_topk "${RETRIEVAL_TOPK}" \
  --retrieval_timeout "${RETRIEVAL_TIMEOUT}" \
  --cp_alpha "${CP_ALPHA}" \
  --cp_k_max "${CP_K_MAX}" \
  --aps_temperature "${APS_TEMPERATURE}" \
  --aps_alpha "${APS_ALPHA}" \
  --aps_min_docs "${APS_MIN_DOCS}" \
  --aps_max_docs "${APS_MAX_DOCS}" \
  --enable_llm_hop_extraction \
  --llm_api_base "${LLM_API_BASE}" \
  --llm_api_key_env "${LLM_API_KEY_ENV}" \
  --llm_model "${LLM_MODEL}" \
  --llm_timeout "${LLM_TIMEOUT}" \
  --llm_temperature "${LLM_TEMPERATURE}" \
  --base_model "${BASE_MODEL}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  "$@"

echo "==== ACI output check ===="
ls -lh "${OUTPUT_DIR}/calibration_dataset.parquet" \
       "${OUTPUT_DIR}/rl_train_dataset.parquet" \
       "${OUTPUT_DIR}/lambda_fixed.json" \
       "${OUTPUT_DIR}/initial_scores.pt"
