#!/bin/bash
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
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.wandb_key" ]; then
  WANDB_API_KEY=$(tr -d '\r\n' < "$HOME/.wandb_key")
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
fi
EXTRA_BINDS=()
if [ -d /opt/amdgpu/share/libdrm ]; then
  EXTRA_BINDS+=(-B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro)
fi
if [ -f /usr/local/lib/libbnxt_re-rdmav34.so ]; then
  EXTRA_BINDS+=(-B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so)
fi
exec singularity exec --rocm --cleanenv --containall --pwd "$REPO" \
  -B /shared_silo/scratch:/shared_silo/scratch:rw -B "$HOME:$HOME:ro" -B /tmp:/tmp:rw \
  -B /usr/share/libdrm:/usr/share/libdrm:ro "${EXTRA_BINDS[@]}" \
  "${NRL_ROCM_IMAGE:-/shared_silo/scratch/containers/primus_v26.2_moefix.sif}" \
  env PYTHONPATH="${NRL_ROCM_RL_PACKAGES:-$REPO/experiments/rocm_rl/packages}:${NRL_ROCM_SFT_PACKAGES:-$REPO/experiments/rocm_sft/packages}:$REPO" \
  PATH="$HOME/.local/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  UV_CACHE_DIR=/tmp/uv-nrl-rocm NEMO_RL_PY_EXECUTABLES_SYSTEM=1 \
  HF_HOME=/shared_silo/scratch/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  VLLM_PLUGINS= VLLM_NO_USAGE_STATS=1 \
  PYTHONUNBUFFERED=1 "$@"
