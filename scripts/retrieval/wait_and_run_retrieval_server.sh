#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_HELPER="$SCRIPT_DIR/get_idle_gpu_ids.sh"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"

if [[ -n "${GPU_IDS:-}" ]]; then
    echo "GPU_IDS was set manually to $GPU_IDS; skipping the wait."
else
    while true; do
        if GPU_IDS="$(bash "$GPU_HELPER" "$REQUIRED_GPUS")"; then
            export GPU_IDS
            echo "Detected idle GPUs: $GPU_IDS"
            break
        else
            status=$?
            if (( status == 2 )); then
                echo "Waiting for ${REQUIRED_GPUS} idle GPUs (checking every ${CHECK_INTERVAL_SECONDS}s)..."
                sleep "$CHECK_INTERVAL_SECONDS"
                continue
            fi

            echo "ERROR: GPU detection script failed (exit code: $status)" >&2
            exit "$status"
        fi
    done
fi

# Start the retrieval server as soon as the requested GPUs are available.
exec bash "$SCRIPT_DIR/run_retrieval_server.sh"
