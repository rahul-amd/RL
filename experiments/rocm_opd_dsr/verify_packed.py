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

"""Compare packed Qwen3 logits and gradients with separate SDPA sequences."""

import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_rl.models.huggingface.common import get_flash_attention_kwargs, pack_sequences


def main() -> None:
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).cuda().train()
    lengths = torch.tensor([113, 257, 61], device="cuda")
    input_ids = torch.randint(0, config.vocab_size, (3, 257), device="cuda")
    outputs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for row, length in zip(input_ids, lengths.tolist()):
            outputs.append(model(row[None, :length]).logits)
        reference = torch.cat(outputs, dim=1)
        reference_loss = reference.float().square().mean()
    reference_loss.backward()
    reference_grads = {
        name: p.grad.detach().clone() for name, p in model.named_parameters()
    }
    reference_logits = reference.detach()
    model.zero_grad(set_to_none=True)
    model.set_attn_implementation("flash_attention_2")
    packed_ids, positions, _ = pack_sequences(
        input_ids, lengths, packed_sequence_size=[3], return_attention_mask=False
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        packed = model(
            packed_ids,
            position_ids=positions,
            attention_mask=torch.ones_like(packed_ids),
            flash_attn_kwargs=get_flash_attention_kwargs(lengths),
        ).logits
        packed_loss = packed.float().square().mean()
    packed_loss.backward()
    torch.testing.assert_close(packed, reference_logits, rtol=0.04, atol=0.025)
    squared_error = 0.0
    squared_reference = 0.0
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
        squared_error += (
            (parameter.grad.float() - reference_grads[name].float())
            .square()
            .sum()
            .item()
        )
        squared_reference += reference_grads[name].float().square().sum().item()
    relative_gradient_error = (squared_error / squared_reference) ** 0.5
    assert relative_gradient_error < 0.03, relative_gradient_error
    report = {
        "sequence_lengths": lengths.tolist(),
        "max_logit_error": (packed.float() - reference_logits.float())
        .abs()
        .max()
        .item(),
        "relative_gradient_error": relative_gradient_error,
        "reference_loss": reference_loss.item(),
        "packed_loss": packed_loss.item(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    Path("experiments/rocm_opd_dsr/packed_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
