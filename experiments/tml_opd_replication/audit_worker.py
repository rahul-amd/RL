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

import hashlib
from pathlib import Path

import ray
import torch
from torch.distributed.tensor import DTensor

from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)


@ray.remote
class AuditedSFTWorker(DTensorPolicyWorkerV2Impl):  # pragma: no cover
    def _init_checkpoint_manager(self, config_updates=None) -> None:
        # Process-based async DCP can collide with a port claimed during training.
        updates = dict(config_updates or {})
        updates["is_async"] = False
        super()._init_checkpoint_manager(config_updates=updates)

    def __init__(self, config, *args, **kwargs):
        torch.manual_seed(config["initialization_seed"])
        super().__init__(config, *args, **kwargs)
        self._check_adapter_coverage()

    def _check_adapter_coverage(self):
        trainable = [
            name for name, p in self.model.named_parameters() if p.requires_grad
        ]
        expected_count = 2 * (7 * self.model.config.num_hidden_layers + 1)
        if len(trainable) != expected_count or any("lora_" not in n for n in trainable):
            raise ValueError(
                f"Expected {expected_count} adapter tensors across all projections, "
                f"found {len(trainable)}. Check fully qualified target patterns."
            )
        if not any("lm_head" in name for name in trainable):
            raise ValueError("Output-head LoRA is missing.")

    def parameter_audit(self):
        self._check_adapter_coverage()
        frozen_hash = hashlib.sha256()
        adapter_hash = hashlib.sha256()
        trainable = {}
        for name, parameter in self.model.named_parameters():
            value = parameter.detach()
            if isinstance(value, DTensor):
                value = value.to_local()
            value = value.cpu().contiguous()
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise ValueError(f"Unexpected trainable base parameter: {name}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite adapter parameter: {name}")
                trainable[name] = list(parameter.shape)
                digest = adapter_hash
            else:
                digest = frozen_hash
            digest.update(name.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return {
            "frozen_sha256": frozen_hash.hexdigest(),
            "adapter_sha256": adapter_hash.hexdigest(),
            "trainable_shapes": trainable,
        }

    def verify_initial_adapter(self):
        from safetensors.torch import load_file

        path = self.cfg["dtensor_cfg"]["lora_cfg"]["restore_from"]
        saved = load_file(str(Path(path) / "adapter_model.safetensors"))
        checked = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = (
                parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            )
            key = "base_model.model." + name
            if key not in saved or not torch.equal(value.detach().cpu(), saved[key]):
                raise ValueError(f"Initial adapter mismatch: {name}")
            checked += 1
        assert checked == len(saved) == 506
        return {"verified_adapter_tensors": checked}
