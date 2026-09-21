#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --job-name=tml-sft-resume
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
PREVIOUS_JOB=${1:?Pass the preceding job ID}
TML_SFT_RUN_ID=${2:?Pass the original full SFT run ID}
shift 2
PREVIOUS_STATE=$(sacct -X -j "$PREVIOUS_JOB" --noheader --parsable2 --format=State)
if [ "$PREVIOUS_STATE" != TIMEOUT ]; then
  echo "Source job $PREVIOUS_JOB ended with $PREVIOUS_STATE; continuation requires TIMEOUT."
  exit 1
fi
TML_SFT_CONFIG=$(uv run --no-project --python ../prime-rl/.venv/bin/python python - "$TML_SFT_RUN_ID" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(f"experiments/tml_opd_replication/run_{sys.argv[1]}/checkpoints")
checkpoints = [p for p in root.glob("step_*") if re.fullmatch(r"step_\d+", p.name)]
if not checkpoints:
    raise FileNotFoundError(f"No completed checkpoint in {root}")
latest = max(checkpoints, key=lambda p: int(p.name.split("_")[1]))
for relative in (
    "config.yaml", "training_info.json", "train_dataloader.pt",
    "policy/weights/model/adapter_model.safetensors",
    "policy/weights/model/adapter_config.json", "policy/optimizer/optim/.metadata",
):
    if not (latest / relative).is_file():
        raise FileNotFoundError(latest / relative)
state = json.loads((latest / "training_info.json").read_text())
if state["total_steps"] >= 3000:
    raise RuntimeError("SFT is already complete")
print(f"Continuing completed checkpoint {latest}, step={state['total_steps']}", file=sys.stderr)
print(latest / "config.yaml")
PY
)
export TML_SFT_RUN_ID TML_SFT_CONFIG
exec bash experiments/tml_opd_replication/sft.sbatch "$@"
