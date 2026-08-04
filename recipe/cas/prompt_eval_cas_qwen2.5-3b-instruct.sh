#!/usr/bin/env bash
set -euo pipefail
set -x

ulimit -n 65535

GPU_IDS="${GPU_IDS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

PROJECT_DIR="$(pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/cas_subset_test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/prompt_eval/cas_qwen2.5-3b-instruct}"

RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8000/retrieve}"
LAMBDA_FIXED_PATH="${LAMBDA_FIXED_PATH:-/home/zixizhu/un_rag/CAS/data/calibration/lambda_fixed.json}"

# Set ENABLE_APS=1 to use APS q_hat from LAMBDA_FIXED_PATH.
# When disabled, retrieval uses topk=3 and no CP/APS post-filtering.
ENABLE_APS="${ENABLE_APS:-0}"

COMMON_ARGS=(
  --generation_backend "${GENERATION_BACKEND:-hf}"
  --model_path "$MODEL_PATH"
  --openai_base_url "${OPENAI_BASE_URL:-http://127.0.0.1:30000}"
  --openai_timeout "${OPENAI_TIMEOUT:-120}"
  --data_path "$TEST_FILE"
  --output_dir "$OUTPUT_DIR"
  --num_workers "${EVAL_NUM_WORKERS:-1}"
  --retrieval_service_url "$RETRIEVAL_SERVICE_URL"
  --lambda_fixed_path "$LAMBDA_FIXED_PATH"
  --topk_no_aps "${TOPK_NO_APS:-3}"
  --topk_aps "${TOPK_APS:-5}"
  --max_assistant_turns "${MAX_ASSISTANT_TURNS:-4}"
  --max_new_tokens_per_turn "${MAX_NEW_TOKENS_PER_TURN:-768}"
  --max_context_tokens "${MAX_CONTEXT_TOKENS:-15000}"
  --temperature "${TEMPERATURE:-0.0}"
  --top_p "${TOP_P:-0.95}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --device_map "${DEVICE_MAP:-auto}"
  --local_files_only "${LOCAL_FILES_ONLY:-1}"
)

if [[ "${OPENAI_MODEL:-}" != "" ]]; then
  COMMON_ARGS+=(--openai_model "$OPENAI_MODEL")
fi

if [[ "${OPENAI_API_KEY:-}" != "" ]]; then
  COMMON_ARGS+=(--openai_api_key "$OPENAI_API_KEY")
fi

if [[ "${MAX_MEMORY:-}" != "" ]]; then
  COMMON_ARGS+=(--max_memory "$MAX_MEMORY")
fi

if [[ "${MAX_SAMPLES:-}" != "" ]]; then
  COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$ENABLE_APS" == "1" ]]; then
  COMMON_ARGS+=(--enable_aps)
fi

python3 "$PROJECT_DIR/scripts/eval_cas_prompt.py" "${COMMON_ARGS[@]}" "$@"
