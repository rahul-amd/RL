# NeMo-RL GRPO and on-policy distillation on ROCm

Cluster smoke recipes using Qwen3-1.7B, short generated arithmetic responses, and three optimizer updates. The distillation recipe uses a frozen Qwen3-4B teacher, teacher top-64 logits, and reverse KL through the native `examples/run_distillation.py` entry point. GRPO uses the native `examples/run_grpo.py`, four responses per prompt, and mathematical answer verification.

## Environment and launch

The scripts use the existing Prime-RL `.venv` interpreter read-only (Python3.12, Torch2.11 ROCm7.2, vLLM0.25.1) in `primus_v26.2_moefix.sif`. No existing virtual environment or container packages are modified. Separate NeMo overlays are loaded from `experiments/rocm_rl/packages` and `experiments/rocm_sft/packages`. The latter is the frozen overlay documented by the SFT smoke recipe.

Install this recipe's additional overlay from the repository root:

```bash
uv pip install --target experiments/rocm_rl/packages --python-version 3.12 --no-deps -r experiments/rocm_rl/requirements-overlay.txt
```

This overlay depends on the existing ROCm environment and the SFT overlay. It is not a replacement for the project's CUDA lockfile. Do not resolve/install CUDA Torch or vLLM wheels over the ROCm environment.

```bash
uv run --no-project --python /usr/bin/python3 experiments/prepare_smoke_data.py
sbatch experiments/rocm_rl/smoke.sbatch grpo
sbatch --job-name=nemo-rocm-opd experiments/rocm_rl/smoke.sbatch distillation
```

Each job requests one MI325X,16CPUs,128GiB RAM on `amd-tw-verification`. Ray and temporary directories are isolated per job. `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` preserves the prepared actor environment. `VLLM_PLUGINS=""` disables unrelated installed vLLM plugins. Training and generation share a GPU and transfer weights through NeMo's IPC path. Prefix caching and CUDA graphs are disabled. No remote logging is enabled.

Outputs: `slurm-JOBID.log`, `run_JOBID/logs/exp_001/`, and `run_JOBID/checkpoints/`.

## ROCm compatibility change

`nemo_rl/utils/nvml.py` obtains GPU UUIDs and free memory from PyTorch on ROCm. NVML remains the NVIDIA path. The former NVML-only helper failed during vLLM device identification. A real GPU probe verifies both new helper results; the existing eight NVML utility tests pass. The best-effort diagnostic logger still prints NVML unavailable on AMD.

## Recorded smoke results

Both jobs completed with exit code 0 on `tus1-p15-g62`:

| Algorithm | Job | Updates | Elapsed | Result |
|---|---|---|---|---|
| GRPO | 49109 | 3 | 3m02s | 48 responses; rewards 0.8125, 1.0, 1.0; generation KL error approximately 0.0002 |
| OPD | 49110 | 3 | 3m51s | 12 responses; losses 0.0730, 0.0997, 0.0431 |

GRPO's first batch had four responses with nonzero advantages. Its next two batches had all-correct rewards and zero group advantages. This easy task verifies that the update path works; those saturated batches do not demonstrate sustained GRPO learning. OPD losses are measured on different student-generated batches, not a fixed validation set.

Both runs saved model, optimizer, tokenizer, and dataloader checkpoints. A CPU DCP load of `model.layers.0.self_attn.q_proj.weight` confirmed finite saved values and actual changes from the initial Hugging Face weights: maximum absolute changes of 1.684e-6 (GRPO) and 2.861e-6 (OPD). This verifies a representative parameter, not every saved tensor or full training resumption. Details are in `results.json`.

These short runs establish execution, teacher scoring, IPC weight refresh, and student updates on ROCm. They do not establish convergence or downstream accuracy. This OPD recipe is native top-k KL distillation; the separate GRPO-style OPD-advantages path has not been tested.
