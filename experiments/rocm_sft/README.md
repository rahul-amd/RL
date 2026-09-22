# NeMo-RL SFT on ROCm

This smoke recipe runs the native `examples/run_sft.py` entry point with the original DTensor/FSDP2 worker (`policy.dtensor_cfg._v2=false`). It uses Qwen3-1.7B, BF16 compute with the worker's FP32 master weights, 64 local arithmetic training examples, 8 validation examples, and 10 optimizer steps. This is an execution check, not a model-quality benchmark.

## Environment

The cluster launcher uses `/shared_silo/scratch/containers/primus_v26.5-pytorch2.12-te2.15.sif` (Python 3.12, PyTorch 2.12.0 ROCm 7.15). A separate package overlay supplies Transformers 5.12.1, Ray 2.55.1, and the remaining NeMo imports. The image and existing virtual environments are unchanged. The root CUDA lockfile is not the environment used by this recipe.

From the repository root, install the overlay:

```bash
uv pip install --target experiments/rocm_sft/packages --python-version 3.12 --no-deps -r experiments/rocm_sft/requirements-overlay.txt
```

`--no-deps` is intentional: the overlay requirements are frozen, while PyTorch and its ROCm libraries come from the image. Resolving the root project would select CUDA dependencies. `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` makes Ray actors reuse the container interpreter and overlay.

## Run

```bash
uv run --no-project --python /usr/bin/python3 experiments/prepare_smoke_data.py
sbatch experiments/rocm_sft/smoke.sbatch
```

The job requests one GPU, 8 CPUs, 128 GiB RAM, and 30 minutes on `amd-tw-verification`. Follow `experiments/rocm_sft/slurm-JOBID.log`. Results and checkpoints go under `experiments/rocm_sft/run_JOBID/`.

For two GPUs, use `sbatch --gpus-per-node=2 experiments/rocm_sft/smoke.sbatch`.
The launcher derives the visible GPU list and NeMo/Ray resource counts from the allocation.

The launcher uses a separate PID namespace and job-specific Ray temporary directory. GPU IDs are local to the Slurm/container allocation; the one visible GPU is index 0. Ray resources are bounded explicitly because host CPU and memory discovery can exceed the Slurm allocation.

The model weights must already exist in `/shared_silo/scratch/hf_cache`. Remote logging and GPU monitoring are disabled. Tensor/context parallelism and sequence packing are disabled.

## Recorded smoke results

Job **49104** completed on `tus1-p15-g62` with exit code 0 in 87 seconds: one MI325X, 10 optimizer steps, and saved model/optimizer/tokenizer/dataloader checkpoints at steps 5 and 10. Initial validation loss was 1.7654, step-5 validation loss approximately 0.0001, and final validation loss 0.315649. The rebound illustrates why this tiny arithmetic task is only an execution check. Training loss decreased from 1.8822 to below 0.0001.

Job **49105** completed on two MI325X GPUs with exit code 0 in 81 seconds. Both FSDP ranks trained for 10 steps and saved their model and optimizer checkpoint shards. Validation loss moved from 1.7654 to 0.690156; all reported losses were finite. The two runs are execution checks, not a numerical-parity claim. See `results.json` for the recorded metrics and verified checkpoint locations.

Both initial jobs requested 64 GiB RAM; Slurm reported approximately 70–72 GiB peak RSS for their batch steps. The launcher now requests 128 GiB for headroom during checkpointing.

No NeMo training source changes were needed. The MI325X theoretical-FLOPS lookup warns that MFU is unavailable; training still completes. This recipe does not establish support for Automodel, Megatron, rollout generation, or OPD.
