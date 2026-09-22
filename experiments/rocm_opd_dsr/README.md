# OPD on Prime-RL's DSR solvable split

These are the recorded Qwen3-1.7B student / Qwen3-32B teacher comparison recipes. See [ROCm setup](../README.md) for the environment. They use native conditional top-k reverse KL, which differs from the sampled-token objective in the [Thinking Machines reproduction](../tml_opd_replication/README.md).

`data_manifest.json` identifies the existing Prime-RL Parquet splits and checksums. Export the 3,004 training and 300 validation rows before launching; adjust only the source paths if the same files moved:

```bash
bash experiments/rocm_rl/container.sh \
  uv run --no-project --python /path/to/rocm/python \
  python experiments/rocm_opd_dsr/prepare_data.py
sbatch experiments/rocm_opd_dsr/train.sbatch
```

`train.sbatch` defaults to `dedicated.yaml`: four GPUs split between two student/teacher ranks and two generation ranks. `NRL_CONFIG_PATH` selects another YAML, and trailing arguments supply Hydra overrides. The default uses k=20, packing, 8 prompts × 16 responses, a 32,768-token generation limit, a 36,864-token sequence budget, AdamW LR 3e-6 and W&B logging. `distillation.yaml` retains the unpacked k=64 comparison; `packed.yaml` enables packing; the smoke YAMLs use shorter generations and smaller teachers.

Sequence packing and dynamic batching cannot both be enabled for the same Policy. Packing already groups variable-length samples into token-budget bins. Dedicated generation avoids the observed ROCm vLLM sleep/wake memory-growth failure. Preparation and finish errors now propagate so a failed worker does not leave a silently stalled training job. RCCL transfers weights between separate NeMo-managed ranks.

`verify_packed.py` checks packed versus independent attention/gradients. `verify_rccl_group.py` checks FP32/BF16 broadcasts across roots and streams, repeated abort, and use after abort. `recover_dataloader.py`, `inspect_live_memory.py` and `resume.sbatch` are recovery tools for the recorded run; inspect their run/checkpoint paths before using them.

The small JSON summaries retain historical measurements and local artifact paths. The initial held-out baseline was 110/128 (85.9375%). First-update times were 17.59 minutes unpacked, 10.23 minutes on two GPUs with packing/k=64, and 7.30 minutes on four GPUs with packing/k=20. These are execution/throughput observations across different generated batches, not controlled convergence comparisons. Historical W&B runs may have been deleted to reclaim storage; all referenced checkpoints and response logs remain local and are not included in this repository.
