#!/usr/bin/env bash
set -euo pipefail

REQUIRED_GPUS="${1:-4}"
IDLE_MAX_MEMORY_MB="${IDLE_MAX_MEMORY_MB:-500}"
IDLE_MAX_UTIL_PERCENT="${IDLE_MAX_UTIL_PERCENT:-10}"

if ! [[ "$REQUIRED_GPUS" =~ ^[0-9]+$ ]] || (( REQUIRED_GPUS <= 0 )); then
    echo "ERROR: REQUIRED_GPUS 必须是正整数，当前值: $REQUIRED_GPUS" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi 不可用，无法检测 GPU 状态" >&2
    exit 1
fi

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

idle_gpu_ids=()
for row in "${gpu_rows[@]}"; do
    IFS=',' read -r idx mem_used gpu_util <<< "$row"
    idx="${idx//[[:space:]]/}"
    mem_used="${mem_used//[[:space:]]/}"
    gpu_util="${gpu_util//[[:space:]]/}"

    if [[ "$mem_used" == "N/A" || "$gpu_util" == "N/A" ]]; then
        continue
    fi

    if [[ "$idx" =~ ^[0-9]+$ && "$mem_used" =~ ^[0-9]+$ && "$gpu_util" =~ ^[0-9]+$ ]]; then
        if (( mem_used <= IDLE_MAX_MEMORY_MB && gpu_util <= IDLE_MAX_UTIL_PERCENT )); then
            idle_gpu_ids+=("$idx")
        fi
    fi
done

if (( ${#idle_gpu_ids[@]} < REQUIRED_GPUS )); then
    # 约定退出码 2 表示“GPU 数量不足，可重试”。
    exit 2
fi

selected_gpu_ids=("${idle_gpu_ids[@]:0:REQUIRED_GPUS}")
(IFS=,; echo "${selected_gpu_ids[*]}")
