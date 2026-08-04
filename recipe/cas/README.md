# CAS recipes

完整操作流程、参数说明和故障排查见仓库根目录的 [`README.md`](../../README.md)。

当前正式支持两条训练链路：

| 模型 | 校准入口 | 训练入口 |
|---|---|---|
| Qwen2.5-3B-Instruct | `prepare_cas_aci_full_qwen2.5-3b.sh` | `train_cas_grpo_qwen2.5-3b-instruct.sh` |
| Qwen3-8B | `prepare_cas_aci_full_qwen3-8b.sh` | `train_cas_grpo_qwen3-8b-instruct.sh` |

所有命令必须从仓库根目录执行。运行训练前，应先完成数据预处理、检索服务部署和对应模型的校准流程。
