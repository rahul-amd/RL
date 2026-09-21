#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=nemo-rocm-grpo
#SBATCH --output=experiments/rocm_rl/slurm-%j.log
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
cd "${NRL_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
TASK_TMP=/tmp/nrr_${SLURM_JOB_ID}
ALGORITHM=${1:-grpo}
if [ "$#" -gt 0 ]; then shift; fi
GPU_COUNT=${SLURM_GPUS_ON_NODE:?}
CPU_COUNT=${SLURM_CPUS_PER_TASK:?}
VISIBLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
mkdir -p "$TASK_TMP"
exec experiments/rocm_rl/container.sh env TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" \
  XDG_CACHE_HOME="$TASK_TMP/cache" TRITON_CACHE_DIR="$TASK_TMP/triton" \
  XDG_CONFIG_HOME="$TASK_TMP/config" MPLCONFIGDIR="$TASK_TMP/matplotlib" \
  WANDB_DATA_DIR="$TASK_TMP/wandb-data" WANDB_CACHE_DIR="$TASK_TMP/wandb-cache" \
  WANDB_CONFIG_DIR="$TASK_TMP/wandb-config" \
  RAY_OVERRIDE_RESOURCES="{\"CPU\":$CPU_COUNT,\"GPU\":$GPU_COUNT,\"memory\":51539607552,\"object_store_memory\":2147483648}" \
  NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  uv run --no-project --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python \
  python "examples/run_${ALGORITHM}.py" --config "experiments/rocm_rl/${ALGORITHM}.yaml" \
  "cluster.gpus_per_node=$GPU_COUNT" \
  "logger.log_dir=experiments/rocm_rl/run_${SLURM_JOB_ID}/logs" \
  "checkpointing.checkpoint_dir=experiments/rocm_rl/run_${SLURM_JOB_ID}/checkpoints" "$@"
