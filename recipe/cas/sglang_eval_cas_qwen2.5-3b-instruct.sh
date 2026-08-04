#!/usr/bin/env bash
set -euo pipefail
set -x

ulimit -n 65535

GPU_IDS="${GPU_IDS:-2,5}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/cas_train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/cas_subset_test.parquet}"

CONFIG_PATH="$PROJECT_DIR/configs"
REWARD_FUNCTION_PATH="$PROJECT_DIR/cas/reward_score"
PROJECT_NAME="${PROJECT_NAME:-CAS}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-CAS_qwen2.5-3b-instruct_sglang_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/prompt_eval/cas_qwen2.5-3b-instruct_sglang}"
DUMP_VALIDATION_DATA_DIR="${DUMP_VALIDATION_DATA_DIR:-$OUTPUT_DIR/validation_data}"

RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8001/retrieve}"
LAMBDA_FIXED_PATH="${LAMBDA_FIXED_PATH:-/home/zixizhu/un_rag/CAS/data/calibration/lambda_fixed.json}"
ENABLE_APS="${ENABLE_APS:-0}"

# CAS uses VERL's existing Search-R1 tag-protocol parser for compatibility.
VERL_TOOL_FORMAT="search_r1"

mkdir -p "$OUTPUT_DIR"
TOOL_CONFIG="$OUTPUT_DIR/search_tool_config_${ENABLE_APS}.yaml"

if [[ "$ENABLE_APS" == "1" ]]; then
  CP_FILTER_MODE="aps"
  TOPK="${TOPK_APS:-5}"
else
  CP_FILTER_MODE="off"
  TOPK="${TOPK_NO_APS:-3}"
fi

cat > "$TOOL_CONFIG" <<EOF
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: $RETRIEVAL_SERVICE_URL
      topk: $TOPK
      num_workers: ${SEARCH_NUM_WORKERS:-120}
      rate_limit: ${SEARCH_RATE_LIMIT:-120}
      timeout: ${SEARCH_TIMEOUT:-30}
      cp_filter_mode: $CP_FILTER_MODE
      static_cp_enable: false
      lambda_fixed_path: $LAMBDA_FIXED_PATH
      lambda_fixed: null
      cp_k_max: ${CP_K_MAX:-10}
      aps_temperature: ${APS_TEMPERATURE:-0.01}
      aps_alpha: ${APS_ALPHA:-0.10}
      aps_q_hat: null
      aps_min_docs: ${APS_MIN_DOCS:-2}
      aps_max_docs: ${APS_MAX_DOCS:-5}
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Searches the web for relevant information based on the given query.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              item:
                type: string
              description: A list of fully-formed semantic queries. The tool will return search results for each query.
          required:
            - query_list
EOF

python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH/train" \
  --config-name="grpo" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.logger='["console"]' \
  trainer.validation_data_dir="$DUMP_VALIDATION_DATA_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-2}" \
  custom_reward_function.path="$REWARD_FUNCTION_PATH/cas_format.py" \
  custom_reward_function.name=compute_score_em \
  +custom_reward_function.reward_kwargs.structure_format_score=0.2 \
  +custom_reward_function.reward_kwargs.final_format_score=0.1 \
  +custom_reward_function.reward_kwargs.retrieval_score=0 \
  +custom_reward_function.reward_kwargs.format_score=0 \
  +custom_reward_function.reward_kwargs.score=1.0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-128}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-2}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.multi_turn.format="$VERL_TOOL_FORMAT" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-4}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ATTENTION_BACKEND:-flashinfer}" \
  "$@"
