#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --job-name=nemo-opd-dsr-resume
#SBATCH --output=experiments/rocm_opd_dsr/slurm-%j.log
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
PREVIOUS_JOB=${1:?Pass the timed-out source job ID}
PREVIOUS_STATE=$(sacct -X -j "$PREVIOUS_JOB" --noheader --parsable2 --format=State)
if [ "$PREVIOUS_STATE" != TIMEOUT ]; then
  echo "Source job $PREVIOUS_JOB ended with $PREVIOUS_STATE; automatic continuation requires TIMEOUT."
  exit 1
fi
CHECKPOINT_DIR="experiments/rocm_opd_dsr/run_${PREVIOUS_JOB}/checkpoints"
LATEST_CHECKPOINT=$(uv run --no-project --python ../prime-rl/.venv/bin/python python - "$CHECKPOINT_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoints = [p for p in root.glob("step_*") if re.fullmatch(r"step_\d+", p.name)]
if not checkpoints:
    raise FileNotFoundError(f"No completed checkpoints in {root}")
latest = max(checkpoints, key=lambda p: int(p.name.split("_")[1]))
for relative in (
    "config.yaml",
    "training_info.json",
    "train_dataloader.pt",
    "policy/weights/.metadata",
    "policy/optimizer/.metadata",
):
    if not (latest / relative).is_file():
        raise FileNotFoundError(latest / relative)
state = json.loads((latest / "training_info.json").read_text())
print(f"Resuming {latest}: {state['total_steps']} completed updates", file=sys.stderr)
if state["total_steps"] >= 150:
    raise RuntimeError("The requested 150 updates are already complete")
print(latest)
PY
)

exec env NRL_CONFIG_PATH="$LATEST_CHECKPOINT/config.yaml" \
  bash experiments/rocm_opd_dsr/train.sbatch \
  "checkpointing.checkpoint_dir=$CHECKPOINT_DIR"
