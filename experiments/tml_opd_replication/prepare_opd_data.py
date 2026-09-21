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

"""Prepare DeepMath prompts with the historical Qwen3 renderer."""

import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
CACHE = Path("/shared_silo/scratch/hf_cache/hub")
source = CACHE / f"datasets--zwhe99--DeepMath-103K/snapshots/{REVISION}/data"
tokenizer = AutoTokenizer.from_pretrained(
    CACHE
    / "models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
)
tokenizer.chat_template = (ROOT / "qwen3_historical.jinja").read_text()
output = ROOT / "data/deepmath_historical.jsonl"
output.parent.mkdir(parents=True, exist_ok=True)
rows, truncated, max_tokens = 0, 0, 0
with output.with_suffix(".tmp").open("w") as stream:
    for shard in sorted(source.glob("*.parquet")):
        for row in pq.read_table(shard, columns=["question"]).to_pylist():
            question = row["question"]
            tokens = tokenizer.encode(question)
            if len(tokens) > 1024:
                question = tokenizer.decode(tokens[:1024])
                truncated += 1
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            assert isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)
            max_tokens = max(max_tokens, len(prompt))
            stream.write(
                json.dumps({"idx": rows, "question": question, "input_ids": prompt})
                + "\n"
            )
            rows += 1
        print("Prepared", rows, "prompts", flush=True)
assert rows == 103022, rows
assert max_tokens + 4096 <= 5184, max_tokens
output.with_suffix(".tmp").replace(output)
manifest = {
    "dataset": "zwhe99/DeepMath-103K",
    "revision": REVISION,
    "rows": rows,
    "truncated": truncated,
    "max_rendered_tokens": max_tokens,
    "order": "original parquet shard and row order",
    "raw_prompt_truncation": 1024,
}
(ROOT / "data/deepmath_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest)
