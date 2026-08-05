# CAS recipes

See the repository-level [`README.md`](../../README.md) for the complete workflow,
configuration options, and troubleshooting guidance.

The following training pipelines are maintained:

| Model | Calibration entry point | Training entry point |
|---|---|---|
| Qwen2.5-3B-Instruct | `prepare_cas_aci_full_qwen2.5-3b.sh` | `train_cas_grpo_qwen2.5-3b-instruct.sh` |
| Qwen3-8B | `prepare_cas_aci_full_qwen3-8b.sh` | `train_cas_grpo_qwen3-8b-instruct.sh` |

Run all commands from the repository root. Before training, complete data
preprocessing, deploy the retrieval service, and calibrate the selected model.
