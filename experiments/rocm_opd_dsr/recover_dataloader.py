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

"""Reconstruct the step-three sampler state and verify every consumed prompt."""

import json
from pathlib import Path

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

if __name__ == "__main__":
    root = Path("experiments/rocm_opd_dsr")
    rows = [json.loads(x) for x in (root / "train.jsonl").read_text().splitlines()]
    by_prompt = {x["input"]: i for i, x in enumerate(rows)}
    assert len(by_prompt) == len(rows)
    torch.manual_seed(42)
    loader = StatefulDataLoader(
        range(len(rows)), batch_size=8, shuffle=True, drop_last=True
    )
    iterator = iter(loader)
    report = []
    for step in range(1, 4):
        expected = []
        for line in (
            (root / f"run_49159/logs/exp_001/train_data_step{step}.jsonl")
            .read_text()
            .splitlines()
        ):
            text = json.loads(line)["content"]
            while isinstance(text, list):
                text = text[0]
            prompt = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            idx = by_prompt[prompt]
            if idx not in expected:
                expected.append(idx)
        actual = next(iterator).tolist()
        print(
            {
                "step": step,
                "saved_prompt_indices": expected,
                "reconstructed_indices": actual,
            },
            flush=True,
        )
        assert actual == expected
        report.append({"step": step, "indices": actual})
    target = root / "run_49159/recovered_step3"
    torch.save(loader.state_dict(), target / "train_dataloader.pt")
    (target / "sampler_recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Sampler state recovered and all 24 consumed prompts match", flush=True)
