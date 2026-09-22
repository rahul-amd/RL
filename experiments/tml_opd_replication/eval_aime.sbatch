#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --job-name=tml-aime24
#SBATCH --array=0-1
#SBATCH --output=experiments/tml_opd_replication/aime-%A_%a.log
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
workers=${SLURM_GPUS_ON_NODE:?}
test "$workers" -ge 1
test "$workers" -le 8
# Slurm's device cgroup exposes allocated GPUs with local contiguous indices.
gpu_devices=$(seq -s, 0 "$((workers - 1))")
models=(sft teacher)
model=${models[$SLURM_ARRAY_TASK_ID]}
output=experiments/tml_opd_replication/run_aime_${SLURM_ARRAY_JOB_ID}
extra=()
if [ -n "${TML_EVAL_ADAPTER:-}" ]; then
  model=opd
  extra=(--adapter "$TML_EVAL_ADAPTER")
fi
echo "Evaluating $model on $SLURM_JOB_NODELIST, GPUs=$SLURM_GPUS_ON_NODE, output=$output"
exec bash experiments/tml_opd_replication/container.sh \
  env CUDA_VISIBLE_DEVICES="$gpu_devices" \
  XDG_CACHE_HOME="/tmp/tml-aime-${SLURM_JOB_ID}/cache" TRITON_CACHE_DIR="/tmp/tml-aime-${SLURM_JOB_ID}/triton" \
  uv run --no-project --python ../prime-rl/.venv/bin/python python \
  experiments/tml_opd_replication/eval_aime.py --model "$model" --output "$output" --workers "$workers" "${extra[@]}"
