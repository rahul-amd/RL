#!/bin/bash
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
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ "${1:-}" != --inside ]; then
  TASK_TMP=/tmp/tml_${SLURM_JOB_ID}
  mkdir -p "$TASK_TMP"
  NODE_IP=$(ip -4 -o addr show dev eno0 | awk '{split($4, parts, "/"); print parts[1]}')
  exec bash experiments/tml_opd_replication/container.sh env \
    TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" \
    XDG_CACHE_HOME="$TASK_TMP/cache" TRITON_CACHE_DIR="$TASK_TMP/triton" \
    XDG_CONFIG_HOME="$TASK_TMP/config" MPLCONFIGDIR="$TASK_TMP/matplotlib" \
    WANDB_DATA_DIR="$TASK_TMP/wandb-data" WANDB_CACHE_DIR="$TASK_TMP/wandb-cache" \
    WANDB_CONFIG_DIR="$TASK_TMP/wandb-config" TOKENIZERS_PARALLELISM=false \
    NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 NCCL_DEBUG=WARN \
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((SLURM_GPUS_ON_NODE - 1))")" \
    TML_NODE_RANK="$SLURM_PROCID" TML_NODE_IP="$NODE_IP" \
    TML_NODE_COUNT="$SLURM_NNODES" TML_GPU_COUNT="$SLURM_GPUS_ON_NODE" \
    TML_CPU_COUNT="$SLURM_CPUS_PER_TASK" \
    TML_RAY_MEMORY="$((SLURM_MEM_PER_NODE * 1024 * 1024 * 3 / 4))" \
    TML_HEAD_IP="$TML_HEAD_IP" TML_RAY_PORT="$TML_RAY_PORT" \
    RAY_ADDRESS="$TML_HEAD_IP:$TML_RAY_PORT" \
    TML_SFT_RUN_ID="$TML_SFT_RUN_ID" TML_SFT_CONFIG="$TML_SFT_CONFIG" \
    bash experiments/tml_opd_replication/multinode_node.sh --inside "$@"
fi
shift
UV=(uv run --no-project --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python)
RAY_ARGS=(--disable-usage-stats --node-ip-address="$TML_NODE_IP"
  --num-cpus="$TML_CPU_COUNT" --num-gpus="$TML_GPU_COUNT"
  --memory="$TML_RAY_MEMORY" --object-store-memory=4294967296)
if [ "$TML_NODE_RANK" -eq 0 ]; then
  "${UV[@]}" python -m ray.scripts.scripts start --head --port="$TML_RAY_PORT" --include-dashboard=false \
    --temp-dir="$RAY_TMPDIR/ray" "${RAY_ARGS[@]}"
  "${UV[@]}" python experiments/tml_opd_replication/multinode_driver.py \
    --config "$TML_SFT_CONFIG" \
    "cluster.num_nodes=$TML_NODE_COUNT" "cluster.gpus_per_node=$TML_GPU_COUNT" \
    "policy.dtensor_cfg.dp_replicate_size=$TML_NODE_COUNT" \
    "logger.log_dir=experiments/tml_opd_replication/run_${TML_SFT_RUN_ID}/logs" \
    "logger.wandb.name=tml-sft-r128-${TML_SFT_RUN_ID}" \
    "logger.wandb.id=tml-sft-r128-${TML_SFT_RUN_ID}" \
    "checkpointing.checkpoint_dir=experiments/tml_opd_replication/run_${TML_SFT_RUN_ID}/checkpoints" "$@"
  touch "experiments/tml_opd_replication/run_${TML_SFT_RUN_ID}/completed_${TML_RAY_PORT}"
else
  "${UV[@]}" python - <<'WAIT'
import os
import socket
import time

address = (os.environ['TML_HEAD_IP'], int(os.environ['TML_RAY_PORT']))
for attempt in range(120):
    try:
        connection = socket.create_connection(address, timeout=2)
    except OSError:
        time.sleep(2)
    else:
        connection.close()
        break
else:
    raise TimeoutError(f'Ray head did not start at {address}')
WAIT
  "${UV[@]}" python -m ray.scripts.scripts start --address="$RAY_ADDRESS" "${RAY_ARGS[@]}"
  COMPLETED="experiments/tml_opd_replication/run_${TML_SFT_RUN_ID}/completed_${TML_RAY_PORT}"
  while [ ! -f "$COMPLETED" ]; do
    sleep 5
  done
fi
