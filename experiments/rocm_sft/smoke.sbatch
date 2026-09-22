#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=nemo-rocm-sft
#SBATCH --output=experiments/rocm_sft/slurm-%j.log
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
TASK_TMP=/tmp/nrl_${SLURM_JOB_ID}
GPU_COUNT=${SLURM_GPUS_ON_NODE:?}
VISIBLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
mkdir -p "$TASK_TMP"
exec experiments/rocm_sft/container.sh env TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" \
  XDG_CACHE_HOME="$TASK_TMP/cache" TRITON_CACHE_DIR="$TASK_TMP/triton" \
  XDG_CONFIG_HOME="$TASK_TMP/config" MPLCONFIGDIR="$TASK_TMP/matplotlib" \
  RAY_OVERRIDE_RESOURCES="{\"CPU\":8,\"GPU\":$GPU_COUNT,\"memory\":51539607552,\"object_store_memory\":2147483648}" \
  NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  uv run --no-project --python /opt/venv/bin/python \
  python examples/run_sft.py --config experiments/rocm_sft/sft.yaml \
  "cluster.gpus_per_node=$GPU_COUNT" \
  "logger.log_dir=experiments/rocm_sft/run_${SLURM_JOB_ID}/logs" \
  "checkpointing.checkpoint_dir=experiments/rocm_sft/run_${SLURM_JOB_ID}/checkpoints" "$@"
