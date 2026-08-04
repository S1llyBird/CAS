# CAS: Conformalized Agentic Search via Adaptive Retrieval and Policy Weighting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Based on Ferret](https://img.shields.io/badge/Based%20on-Ferret-orange.svg)](https://github.com/Tree-Shu-Zhao/ferret)

Official implementation of **CAS**, built on [VERL](https://github.com/volcengine/verl).

This repository is derived from and contains modified code from [Tree-Shu-Zhao/ferret](https://github.com/Tree-Shu-Zhao/ferret), distributed under the Apache License 2.0.

## Environment Setup

Training and retrieval use two separate environments:

| Process | Environment |
|---|---|
| Data preprocessing, calibration, and CAS training | `uv` environment, Python 3.12+, CUDA 12.9+ |
| Retrieval server | Conda environment, Python 3.10, PyTorch 2.4, CUDA 12.6, FAISS-GPU 1.8 |

All commands should be executed from the repository root unless noted otherwise.

### Training environment

```bash
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

### Retrieval environment

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install pytorch==2.4.0 pytorch-cuda=12.6 -c pytorch -c nvidia -y
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

pip install transformers datasets pyserini
pip install uvicorn fastapi scipy==1.11.2 huggingface_hub
```

## Quick Start

### 1. Prepare the training data

Run in the **training environment**:

```bash
source .venv/bin/activate
bash recipe/cas/data_preprocess_cas.sh
```

The processed files are saved under `data/`.

### 2. Prepare and start the retrieval server

Run in the **retrieval environment**.

Download the E5 index and Wikipedia corpus:

```bash
conda activate retriever

export DATA_DIR=/mnt/data/retrieval-corpus
mkdir -p "$DATA_DIR"

hf download --repo-type dataset --local-dir "$DATA_DIR" PeterJinGo/wiki-18-e5-index
hf download --repo-type dataset --local-dir "$DATA_DIR" PeterJinGo/wiki-18-corpus

cd "$DATA_DIR"
cat part_* > e5_Flat.index
gzip -d wiki-18.jsonl.gz
```

Return to the repository root and start the service:

```bash
cd /path/to/CAS
conda activate retriever

DATA_DIR=/mnt/data/retrieval-corpus \
GPU_IDS=6,7 \
bash scripts/retrieval/run_retrieval_server.sh
```

The service listens on `http://127.0.0.1:8000/retrieve` by default. Keep it running in a separate terminal during calibration and training.

### 3. Build calibration artifacts

Return to the **training environment** while the retrieval server remains active:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=<your_api_key>

CUDA_VISIBLE_DEVICES=0,1 \
bash recipe/cas/prepare_cas_aci_full_qwen2.5-3b.sh
```

The outputs are saved to `data/calibration/`:

```text
calibration_dataset.parquet
rl_train_dataset.parquet
lambda_fixed.json
initial_scores.pt
```

Use different GPUs for the retrieval server and calibration model when running them on the same machine.

### 4. Train CAS

Run in the **training environment** with the retrieval server still active:

```bash
source .venv/bin/activate

GPU_IDS=0,1,2,3 \
bash recipe/cas/train_cas_grpo_qwen2.5-3b-instruct.sh
```

Hydra overrides can be appended directly:

```bash
GPU_IDS=0,1 \
bash recipe/cas/train_cas_grpo_qwen2.5-3b-instruct.sh \
  trainer.n_gpus_per_node=2 \
  trainer.logger='["console"]'
```

Checkpoints are written to `checkpoints/CAS/`.

## Evaluation

```bash
ENABLE_APS=1 bash recipe/cas/prompt_eval_cas_qwen2.5-3b-instruct.sh
```

For SGLang/VERL evaluation:

```bash
ENABLE_APS=1 bash recipe/cas/sglang_eval_cas_qwen2.5-3b-instruct.sh
```

## Acknowledgements

CAS is developed from [Ferret](https://github.com/Tree-Shu-Zhao/ferret) and implemented on top of [VERL](https://github.com/volcengine/verl). We thank the authors and contributors of both projects.
