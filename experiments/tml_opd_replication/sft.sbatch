#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --job-name=tml-sft-qwen3-8b
#SBATCH --output=experiments/tml_opd_replication/sft-%j.log
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
TASK_TMP=/tmp/tml_${SLURM_JOB_ID}
SFT_RUN_ID=${TML_SFT_RUN_ID:-$SLURM_JOB_ID}
SFT_CONFIG=${TML_SFT_CONFIG:-experiments/tml_opd_replication/sft.yaml}
mkdir -p "experiments/tml_opd_replication/run_${SFT_RUN_ID}"
exec 9>"experiments/tml_opd_replication/run_${SFT_RUN_ID}/training.lock"
echo "Waiting for run ${SFT_RUN_ID} writer lock"
flock 9
echo "Acquired run ${SFT_RUN_ID} writer lock"
GPU_COUNT=${SLURM_GPUS_ON_NODE:?}
CPU_COUNT=${SLURM_CPUS_PER_TASK:?}
RAY_MEMORY=$((SLURM_MEM_PER_NODE * 1024 * 1024 * 3 / 4))
VISIBLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
mkdir -p "$TASK_TMP"
echo "Allocated $GPU_COUNT GPUs and $CPU_COUNT CPUs on $SLURM_JOB_NODELIST"
exec bash experiments/tml_opd_replication/container.sh env \
  TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" \
  XDG_CACHE_HOME="$TASK_TMP/cache" TRITON_CACHE_DIR="$TASK_TMP/triton" \
  XDG_CONFIG_HOME="$TASK_TMP/config" MPLCONFIGDIR="$TASK_TMP/matplotlib" \
  WANDB_DATA_DIR="$TASK_TMP/wandb-data" WANDB_CACHE_DIR="$TASK_TMP/wandb-cache" \
  WANDB_CONFIG_DIR="$TASK_TMP/wandb-config" TOKENIZERS_PARALLELISM=false \
  RAY_OVERRIDE_RESOURCES="{\"CPU\":$CPU_COUNT,\"GPU\":$GPU_COUNT,\"memory\":$RAY_MEMORY,\"object_store_memory\":4294967296}" \
  NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  uv run --no-project \
  --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python \
  python experiments/tml_opd_replication/run_sft.py \
  --config "$SFT_CONFIG" \
  "cluster.gpus_per_node=$GPU_COUNT" \
  "logger.log_dir=experiments/tml_opd_replication/run_${SFT_RUN_ID}/logs" \
  "logger.wandb.name=tml-sft-r128-${SFT_RUN_ID}" \
  "logger.wandb.id=tml-sft-r128-${SFT_RUN_ID}" \
  "checkpointing.checkpoint_dir=experiments/tml_opd_replication/run_${SFT_RUN_ID}/checkpoints" "$@"
