set -x

ulimit -n 65535

GPU_IDS=${GPU_IDS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=$GPU_IDS

PROJECT_DIR="$(pwd)"

# Setup custom metrics (automatically detects metrics from reward function)
echo "Setting up custom metrics for automatic detection..."
bash "$PROJECT_DIR/scripts/setup_custom_metrics.sh"
DATA_DIR="$PROJECT_DIR/data"
CALIB_DIR=${CALIB_DIR:-"$DATA_DIR/calibration"}
TRAIN_FILE=${TRAIN_FILE:-"$CALIB_DIR/rl_train_dataset.parquet"}
VAL_FILE=${VAL_FILE:-"$DATA_DIR/cas_test.parquet"}
INITIAL_SCORES_PATH=${INITIAL_SCORES_PATH:-"$CALIB_DIR/initial_scores.pt"}
LAMBDA_FIXED_PATH=${LAMBDA_FIXED_PATH:-"$CALIB_DIR/lambda_fixed.json"}

CONFIG_PATH="$PROJECT_DIR/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/cas/reward_score"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
PROJECT_NAME="CAS"
EXPERIMENT_NAME="CAS_qwen2.5-3b-instruct_grpo"

# CAS uses VERL's existing Search-R1 tag-protocol parser for compatibility.
VERL_TOOL_FORMAT="search_r1"

if [[ ! -f "$TRAIN_FILE" || ! -f "$INITIAL_SCORES_PATH" || ! -f "$LAMBDA_FIXED_PATH" ]]; then
    echo "[ERROR] Missing CAS calibration artifacts under: $CALIB_DIR" >&2
    echo "Run recipe/cas/prepare_cas_aci_full_qwen2.5-3b.sh first." >&2
    exit 1
fi

export CAS_LAMBDA_FIXED_PATH="$LAMBDA_FIXED_PATH"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="grpo" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/cas_format.py" \
    custom_reward_function.name=compute_score_em \
    +custom_reward_function.reward_kwargs.structure_format_score=0.2 \
    +custom_reward_function.reward_kwargs.final_format_score=0.1 \
    +custom_reward_function.reward_kwargs.retrieval_score=0 \
    +custom_reward_function.reward_kwargs.format_score=0 \
    +custom_reward_function.reward_kwargs.score=1.0 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    critic.model.path=$BASE_MODEL \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    algorithm.dcr_enable=True \
    algorithm.dcr_initial_scores_path="$INITIAL_SCORES_PATH" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE"  \
    actor_rollout_ref.rollout.multi_turn.format="$VERL_TOOL_FORMAT" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer $@
