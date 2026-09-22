# GPU sanity checks — September 22, 2026

All checks completed on AMD MI325X GPUs in `amd-tw-verification` / normal QoS. These runs use the PR checkout based on `e15134149`, its Automodel pin `72daceff`, and the checkpoint/launcher fixes in this commit. The exact tested source hashes, checkpoint paths, losses and parameter checks are recorded in [gpu-sanity-results.json](gpu-sanity-results.json).

| Check | Job | GPUs | Completed updates | Wall time |
| --- | --- | --- | --- | --- |
| Native SFT | [50285](https://wandb.ai/rahular/opd-tests/runs/0sajexuh) | 1 | 10 | 00:01:27 |
| GRPO | [50286](https://wandb.ai/rahular/opd-tests/runs/q5z4h7ap) | 1 | 3 | 00:02:56 |
| Top-k OPD | [50287](https://wandb.ai/rahular/opd-tests/runs/37nxdibe) | 1 | 3 | 00:03:28 |
| Packed LoRA SFT | [50288](https://wandb.ai/rahular/opd-tests/runs/tml-sft-r128-50288) | 2 | 2 | 00:02:20 |
| LoRA SFT resume | [50290](https://wandb.ai/rahular/opd-tests/runs/tml-sft-r128-50288) | 2 | 2 → 3 | 00:02:12 |
| Sampled-token LoRA OPD | [50289](https://wandb.ai/rahular/opd-tests/runs/tml-opd-50289) | 4 | 2 | 00:04:58 |
| LoRA OPD resume | [50295](https://wandb.ai/rahular/opd-tests/runs/tml-opd-50289) | 4 | 2 → 3 | 00:04:51 |
| Packed top-k OPD, dedicated inference | [50294](https://wandb.ai/rahular/opd-tests/runs/nemo-opd-dsr-50294) | 4 | 6 | 00:03:50 |

Native smoke checks used Qwen3-1.7B, with Qwen3-4B as the OPD teacher. LoRA checks used Qwen3-8B-Base with the Qwen3-8B teacher.

The separate two-GPU RCCL probe (job 50279) passed eight FP32/BF16 broadcasts per rank with rotating roots and a separate CUDA stream, repeated abort, and rejection of a broadcast after abort.

## What was verified

- Every training/resume job exited `COMPLETED`, code `0:0`, and saved complete model/adapter, optimizer and dataloader checkpoints.
- CPU loads of a representative attention projection from native SFT, GRPO, top-k OPD and packed top-k OPD checkpoints confirmed finite values and real changes from the initial model. This samples one parameter tensor, not every full-model tensor.
- Both LoRA recipes audited every parameter shard: frozen base hashes stayed equal, adapters stayed finite and changed. OPD verified all 506 initial adapter tensors per training rank against the SFT checkpoint.
- Both LoRA resumes restored every audited parameter shard exactly, loaded optimizer/dataloader state, completed update 3 and saved another checkpoint.
- W&B logging was enabled for all training jobs under `opd-tests`, group `pr-amd-sanity`. Raw responses and checkpoints remain local.

## Scope and reproduction

Use the environment and overlay setup in [README.md](README.md). The native SFT, GRPO and top-k OPD checks used their checked-in smoke recipes. The dedicated packed check used `NRL_CONFIG_PATH=experiments/rocm_opd_dsr/dedicated_smoke.yaml` with `train.sbatch`.

The LoRA SFT check used `sft.sbatch --audit` on two GPUs with `sft.max_num_steps=2`, `policy.train_global_batch_size=4`, `data.sample_limit=16`, `data.require_complete=false`, and `checkpointing.save_period=1`. It retained the original 16,384-token data and packing budget. The tokenized-data path pointed to the existing full prepared OpenThoughts3 cache.

The LoRA OPD check used `opd.sbatch --audit` on four GPUs (two training/teacher, two dedicated inference), with `distillation.max_num_steps=2`, `distillation.num_prompts_per_step=4`, `distillation.num_generations_per_prompt=2`, `policy.train_global_batch_size=8`, `policy.generation.max_new_tokens=256`, `policy.generation.colocated.resources.gpus_per_node=2`, `policy.generation.vllm_cfg.gpu_memory_utilization=0.4`, and `checkpointing.save_period=1`. It used the existing DeepMath tokenized data and original SFT adapter. Both resumes loaded the saved step-2 config and checkpoint directory, set the step limit to 3, and supplied `--verify-restore-audit` pointing to the prior parameter audit.

These are single-node execution checks with small batches and short OPD generations. They do not rerun full training, validate multi-node scaling, or measure downstream quality. Native GRPO rewards saturated at 1.0 after its first update; checkpoint changes confirm the update path, not sustained learning on this tiny arithmetic dataset.

## Fixes found by the checks

1. Set the native DTensor V1 checkpoint format to `null` in `policy.dtensor_cfg.checkpoint`, and remove the unsupported top-level checkpoint-format field.
2. Match the current DTensor V2 checkpoint-manager signature while keeping the experiment worker’s synchronous saves.
3. Use `++logger.wandb.id` so the dedicated launcher also accepts smoke configs without a predeclared run ID.

No optimizer or reduction dtype settings were changed.
