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

import json

import pytest

from experiments.tml_opd_replication.monitor_sft import (
    completed_step,
    infrastructure_failure,
    latest_checkpoint,
)


def test_progress_requires_a_completed_update():
    log = "Step 50/3000\nTotal step time: 25.35s\nStep 51/3000\nTaking a training step"
    assert completed_step(log) == 50
    assert completed_step("Step 51/3000\nTaking a training step") is None


def test_infrastructure_retry_excludes_memory_and_numerical_errors():
    assert infrastructure_failure(
        "Checkpoint background process failed during initialization"
    )
    assert infrastructure_failure("ActorDiedError: worker exited")
    assert not infrastructure_failure("ActorDiedError: CUDA out of memory")
    assert not infrastructure_failure("Nonfinite adapter parameter")
    assert not infrastructure_failure("ValueError: shape mismatch")


def test_checkpoint_selection_ignores_incomplete_and_requires_optimizer(tmp_path):
    completed = tmp_path / "step_2"
    files = [
        "config.yaml",
        "train_dataloader.pt",
        "policy/weights/model/adapter_model.safetensors",
        "policy/weights/model/adapter_config.json",
        "policy/optimizer/optim/.metadata",
    ]
    for name in files:
        p = completed / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    (completed / "training_info.json").write_text(json.dumps({"total_steps": 2}))
    (tmp_path / "tmp_step_50").mkdir()
    assert latest_checkpoint(tmp_path) == (completed, 2)
    (completed / "policy/optimizer/optim/.metadata").unlink()
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(tmp_path)
