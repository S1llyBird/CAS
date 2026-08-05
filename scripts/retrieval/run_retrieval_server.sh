#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"

if [[ -z "${GPU_IDS:-}" ]]; then
    GPU_HELPER="$SCRIPT_DIR/get_idle_gpu_ids.sh"
    if GPU_IDS="$(bash "$GPU_HELPER" "$REQUIRED_GPUS")"; then
        :
    else
        status=$?
        if (( status == 2 )); then
            echo "ERROR: Could not find ${REQUIRED_GPUS} idle GPUs. Set GPU_IDS manually, for example GPU_IDS=0,1,2,3" >&2
        else
            echo "ERROR: GPU detection script failed (exit code: $status)" >&2
        fi
        exit "$status"
    fi
fi
GPU_IDS="${GPU_IDS//[[:space:]]/}"

file_path="$DATA_DIR"
index_file="$file_path/e5_Flat.index"
corpus_file="$file_path/wiki-18.jsonl"
port=8000

retriever_name=e5
retriever_path=intfloat/e5-base-v2

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Using GPU_IDS=$GPU_IDS"

python "$PROJECT_DIR/cas/retrieval/retrieval_server.py" \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk 3 \
    --retriever_name "$retriever_name" \
    --retriever_model "$retriever_path" \
    --port "$port" \
    --faiss_gpu
