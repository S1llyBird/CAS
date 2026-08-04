#!/usr/bin/env bash
set -euo pipefail
set -x

ulimit -n 65535

# 固定使用 0,1 两张显卡
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data"

# 输入数据（原始 RL 训练集）
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/cas_train.parquet}"

# 产物输出目录（会生成 calibration_dataset / rl_train_dataset / lambda_fixed / initial_scores）
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/calibration}"

# 检索服务配置
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-20}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-30}"

# ACI 基座模型（用于生成 initial_scores.pt）
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# 校准超参数（可按需覆盖）
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-150}"
CP_ALPHA="${CP_ALPHA:-0.20}"
CP_K_MAX="${CP_K_MAX:-10}"
APS_TEMPERATURE="${APS_TEMPERATURE:-0.01}"
APS_ALPHA="${APS_ALPHA:-0.10}"
APS_MIN_DOCS="${APS_MIN_DOCS:-2}"
APS_MAX_DOCS="${APS_MAX_DOCS:-5}"
SEED="${SEED:-42}"

# Hotpot 轨迹缺失时需要 teacher-LLM 补 hop；默认启用
LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
LLM_MODEL="${LLM_MODEL:-deepseek-chat}"
LLM_TIMEOUT="${LLM_TIMEOUT:-60}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_API_KEY_ENV="${LLM_API_KEY_ENV:-OPENAI_API_KEY}"

# 明确检查 API Key，避免运行到中途失败
if [[ -z "${!LLM_API_KEY_ENV:-}" ]]; then
  echo "[ERROR] 环境变量 ${LLM_API_KEY_ENV} 未设置，无法进行 --enable_llm_hop_extraction。"
  echo "请先执行：export ${LLM_API_KEY_ENV}=<your_api_key>"
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

echo "==== ACI 产物检查 ===="
ls -lh "${OUTPUT_DIR}/calibration_dataset.parquet" \
       "${OUTPUT_DIR}/rl_train_dataset.parquet" \
       "${OUTPUT_DIR}/lambda_fixed.json" \
       "${OUTPUT_DIR}/initial_scores.pt"
