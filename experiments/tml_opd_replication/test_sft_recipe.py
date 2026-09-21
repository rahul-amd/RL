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

import ast
import json
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from experiments.tml_opd_replication.sft_data import (
    HistoricalSFTDataset,
    render_conversation,
)
from experiments.tml_opd_replication.sft_loss import SummedSFTLoss
from nemo_rl.algorithms.sft import prepare_sft_batch
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

ROOT = Path(__file__).parent


@pytest.fixture(scope="module")
def tokenizer():
    recipe = json.loads((ROOT / "sft_recipe.json").read_text())
    return AutoTokenizer.from_pretrained(recipe["model_snapshot"])


@pytest.fixture(scope="module")
def historical_renderer(tokenizer):
    source = ROOT / "reference_330b73d/tinker_cookbook/renderers.py"
    tree = ast.parse(source.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"TrainOnWhat", "build_supervised_example"}
    ]
    qwen = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3Renderer"
    )
    selected.append(
        next(
            n
            for n in qwen.body
            if isinstance(n, ast.FunctionDef) and n.name == "_render_message"
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "StrEnum": StrEnum}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace, SimpleNamespace(tokenizer=tokenizer)


@pytest.mark.parametrize("max_length", [16, 16384])
def test_tokens_and_masks_match_historical_renderer(
    tokenizer, historical_renderer, max_length
):
    namespace, renderer = historical_renderer
    cases = [
        [
            {"from": "human", "value": "Solve 3×4.\nGive reasons."},
            {"from": "gpt", "value": "Twelve. </think>\n\\boxed{12}"},
        ],
        [
            {"from": "human", "value": "Why?"},
            {"from": "gpt", "value": "<think>First\n</think>\nYes."},
            {"from": "human", "value": "Explain."},
            {"from": "gpt", "value": "Because."},
        ],
    ]
    for conversation in cases:
        messages = [
            {
                "role": "user" if m["from"] == "human" else "assistant",
                "content": m["value"],
            }
            for m in conversation
        ]
        tokens, mask = namespace["build_supervised_example"](
            [],
            lambda i, m: namespace["_render_message"](renderer, i, m),
            messages,
            namespace["TrainOnWhat"].ALL_ASSISTANT_MESSAGES,
        )
        actual = render_conversation(conversation, tokenizer, max_length)
        assert actual["input_ids"] == tokens[:max_length].tolist()
        assert actual["loss_mask"] == mask[:max_length].tolist()
        template = (ROOT / "qwen3_historical.jinja").read_text()
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert rendered == tokenizer.decode(tokens.tolist())


def test_nemo_collation_preserves_masks(tmp_path, tokenizer):
    row = render_conversation(
        [
            {"from": "human", "value": "3+4?"},
            {"from": "gpt", "value": "<think>Adding.</think> Seven."},
        ],
        tokenizer,
        16384,
    )
    dataset = Dataset.from_list([row])
    dataset.save_to_disk(str(tmp_path / "source"))
    (tmp_path / "tokens-00000.arrow").symlink_to(
        next((tmp_path / "source").glob("*.arrow"))
    )
    actual = prepare_sft_batch(
        rl_collate_fn([HistoricalSFTDataset(tmp_path)[0]]),
        tokenizer=tokenizer,
        only_unmask_final=False,
        make_sequence_length_divisible_by=1,
    )
    assert actual["input_ids"][0].tolist() == row["input_ids"]
    assert actual["token_mask"][0].tolist() == row["loss_mask"]


def test_summed_loss_matches_masked_cross_entropy():
    logits = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    data = BatchedDataDict(
        token_mask=torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]),
        sample_mask=torch.ones(2),
    )
    expected = -(logp * data["token_mask"][:, 1:]).sum()
    actual, metrics = SummedSFTLoss()(logp, data, torch.tensor(2), torch.tensor(5))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )
    assert metrics["loss"] == pytest.approx(expected.item() / 5)


def test_lora_patterns_cover_every_projection():
    import yaml
    from nemo_automodel.components._peft.module_matcher import ModuleMatcher

    config = yaml.safe_load((ROOT / "sft.yaml").read_text())
    lora = config["policy"]["dtensor_cfg"]["lora_cfg"]
    matcher = ModuleMatcher(target_modules=lora["target_modules"])
    linear = torch.nn.Linear(4, 4)
    for layer in range(36):
        for group, names in [
            ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            ("mlp", ["gate_proj", "up_proj", "down_proj"]),
        ]:
            for name in names:
                assert matcher.match(linear, f"model.layers.{layer}.{group}.{name}")
    assert matcher.match(linear, "lm_head")
