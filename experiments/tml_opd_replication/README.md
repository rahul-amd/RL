# Thinking Machines LoRA OPD reproduction on ROCm

The completed recipe, provenance, assumptions and results are in [TM_OPD_RECIPE.md](../../TM_OPD_RECIPE.md). The local Qwen3-8B-Base rank-128 SFT adapter scored 54.5833% on AIME24; continuing that adapter for 100 OPD updates reached 61.25%. The frozen Qwen3-8B teacher scored 76.25%. These use 960 samples per model/checkpoint and a locally specified evaluator; the historical evaluation protocol was not fully published.

See [ROCm setup](../README.md) before launching. All commands run from the repository root. `sft_recipe.json` and `manifest.json` pin the model/dataset revisions, historical cookbook commit and local assumptions. No Tinker service or API key is needed.

## Data and SFT

Cache the pinned Qwen3-8B-Base model and all 120 OpenThoughts3 parquet shards at the paths in `sft_recipe.json`. Adjust cache locations there and in the YAML files for your installation. The preparation job materializes the first 384,000 examples from the historical seed-0 streaming shuffle, renders the historical Qwen3 chat format, preserves assistant-body masks and truncates at 16,384 tokens. It fails if the output directory already exists or the input is incomplete.

```bash
sbatch experiments/tml_opd_replication/prepare_data.sbatch
# After data preparation completes:
sbatch experiments/tml_opd_replication/sft_multinode.sbatch
```

`sft_multinode.sbatch` requests 4 nodes × 8 GPUs, using eight FSDP shards per node and four replicas. `sft.sbatch` is the single-node launcher. Training uses 3,000 updates, batch 128, rank-128 LoRA on attention/MLP/lm_head, summed token gradients, Adam LR 1e-3 with linear decay, and sequence packing. `audit_worker.py` provides synchronous checkpoint saving to avoid the observed async-save rendezvous failure, plus optional frozen-base, adapter-change and exact-restore audits. Checkpoints include optimizer and dataloader state every five updates.

## OPD continuation

Cache the DeepMath-103K revision in `manifest.json`, then prepare prompts:

```bash
bash experiments/tml_opd_replication/container.sh \
  uv run --no-project --python /path/to/rocm/python \
  python experiments/tml_opd_replication/prepare_opd_data.py
```

Set `policy.dtensor_cfg.lora_cfg.restore_from` in `opd.yaml` to the completed SFT checkpoint's adapter directory. The checked-in value identifies the original local run. Also update any cache and checkpoint paths if the checkout moved. Continue the same adapter with a fresh optimizer; leave the frozen base unchanged.

```bash
sbatch experiments/tml_opd_replication/opd_multinode.sbatch
```

The launcher requests 32 GPUs: 16 for student training/teacher scoring and 16 for dedicated vLLM generation. `opd.sbatch` supports single-node execution checks. The objective is sampled-token reverse KL with detached teacher-minus-rollout log-probability advantages, unclipped importance weighting and **summed** gradients. Each update uses 512 prompts × 4 responses, 4,096 new tokens, unfiltered temperature-1 sampling, constant Adam LR 1e-4 and one update per rollout batch. Packing groups variable-length sequences; the mutually exclusive dynamic-batching switch is disabled.

## Evaluation and monitoring

`eval_aime.sbatch` / `eval_aime.py` evaluate SFT adapters, OPD adapters or the teacher. The launcher defaults to the recorded SFT path; inspect and override its `TML_EVAL_*` variables before reuse. It validates all adapter tensors, checks FP32 logits across PEFT merge, exports a separate BF16 inference model, and uses the actual allocated worker count. The scoring protocol is 30 AIME24 problems × 32 samples, temperature 0.6, top-p 0.95, top-k 20 and a 32,768-token total context. Full responses stay local; W&B receives summaries.

`monitor_sft.py` and `monitor_opd.py` accept `--config /path/to/monitor_config.json`. They acquire an exclusive monitor lock, inspect Slurm and complete checkpoints, retry recognized infrastructure failures, and stop on unknown/numerical failures or external cancellation. They can cancel stalled jobs and submit replacements; initialize their state with the jobs belonging to your run. The OPD monitor additionally submits evaluation at configured checkpoints. Historical monitor state is deliberately not distributed because its job IDs and output paths belong to completed runs. The monitor config/state examples describe the required fields; replace the run/job IDs with integers, copy the state example to the configured state path, and point `--config` at your edited config before starting either monitor. Start after a complete checkpoint exists; a restart requires optimizer state.

## Verification

The existing tests cover historical renderer/mask parity, summed-loss gradients, LoRA target coverage, sampled reverse-KL gradients and accumulation, prompt/log-probability alignment, and checkpoint/retry selection. Fetch the checksummed historical renderer before running them:

```bash
uv run --no-project --python /usr/bin/python3 experiments/tml_opd_replication/prepare_reference.py
bash experiments/tml_opd_replication/container.sh \
  uv run --no-project --python /path/to/rocm/python python -m pytest \
  experiments/tml_opd_replication/test_sft_recipe.py \
  experiments/tml_opd_replication/test_opd_loss.py \
  experiments/tml_opd_replication/test_sft_monitor.py -q
```

Tests require the pinned tokenizer cache, package overlays and Automodel source. `sft_validation.json` and `opd_validation.json` preserve the recorded GPU execution/audit results. Generated data, reference source, dependencies, checkpoints and logs are ignored by Git.
