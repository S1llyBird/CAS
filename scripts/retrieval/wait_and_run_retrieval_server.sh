#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_HELPER="$SCRIPT_DIR/get_idle_gpu_ids.sh"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"

if [[ -n "${GPU_IDS:-}" ]]; then
    echo "GPU_IDS 已手动指定为 $GPU_IDS，跳过等待。"
else
    while true; do
        if GPU_IDS="$(bash "$GPU_HELPER" "$REQUIRED_GPUS")"; then
            export GPU_IDS
            echo "检测到空闲 GPU: $GPU_IDS"
            break
        else
            status=$?
            if (( status == 2 )); then
                echo "等待 ${REQUIRED_GPUS} 张空闲 GPU（每 ${CHECK_INTERVAL_SECONDS}s 检查一次）..."
                sleep "$CHECK_INTERVAL_SECONDS"
                continue
            fi

            echo "ERROR: GPU 检测脚本执行失败（exit code: $status）" >&2
            exit "$status"
        fi
    done
fi

# 用户要求检测到 GPU 后只启动这个脚本。
exec bash "$SCRIPT_DIR/run_retrieval_server.sh"
