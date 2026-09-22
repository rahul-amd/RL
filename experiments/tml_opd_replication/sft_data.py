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

"""Historical Qwen3 rendering and pretokenized NeMo SFT data."""

from itertools import groupby
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets


def render_conversation(conversations, tokenizer, max_length):
    tokens, mask = [], []
    for index, message in enumerate(conversations):
        role = "user" if message["from"] == "human" else "assistant"
        content = message["value"]
        header = ("\n" if index else "") + f"<|im_start|>{role}\n"
        if role == "assistant" and "<think>" not in content:
            header += "<think>\n"
        header_tokens = tokenizer.encode(header, add_special_tokens=False)
        body_tokens = tokenizer.encode(content + "<|im_end|>", add_special_tokens=False)
        tokens.extend(header_tokens)
        mask.extend([0] * len(header_tokens))
        tokens.extend(body_tokens)
        # The historical renderer always trains the final message's body.
        train_body = role == "assistant" or index == len(conversations) - 1
        mask.extend([int(train_body)] * len(body_tokens))
    if len(tokens) < 2:
        raise ValueError("A training example must contain at least two tokens.")
    return {"input_ids": tokens[:max_length], "loss_mask": mask[:max_length]}


class HistoricalSFTDataset(torch.utils.data.Dataset):
    def __init__(self, directory, limit=None):
        paths = sorted(Path(directory).glob("tokens-*.arrow"))
        if not paths:
            raise FileNotFoundError(f"No token shards in {directory}")
        self.data = concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
        if limit is not None:
            if limit > len(self.data):
                raise ValueError(
                    f"Requested {limit} rows, only {len(self.data)} available"
                )
            self.data = self.data.select(range(limit))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        messages = []
        offset = 0
        for value, values in groupby(row["loss_mask"]):
            length = sum(1 for _ in values)
            messages.append(
                {
                    "role": "assistant" if value else "user",
                    "content": "",
                    "token_ids": torch.tensor(
                        row["input_ids"][offset : offset + length], dtype=torch.long
                    ),
                }
            )
            offset += length
        return {
            "message_log": messages,
            "length": len(row["input_ids"]),
            "loss_multiplier": 1.0,
            "extra_env_info": None,
            "idx": index,
            "task_name": "openthoughts3",
        }
