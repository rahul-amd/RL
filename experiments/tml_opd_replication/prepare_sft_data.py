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

"""Materialize the historical shuffled SFT stream, without a second shuffle."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import datasets
from datasets import Features, List, Value, load_dataset
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer

from experiments.tml_opd_replication.sft_data import render_conversation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    output = Path(recipe["tokenized_data"])
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(Path(recipe["dataset_snapshot"]).glob("data/*.parquet"))
    if len(files) != 120:
        raise ValueError(f"Expected 120 OpenThoughts3 shards, found {len(files)}")
    stream = load_dataset(
        "parquet", data_files=[str(p) for p in files], split="train", streaming=True
    ).shuffle(seed=recipe["shuffle_seed"], buffer_size=recipe["shuffle_buffer"])
    stream.set_epoch(0)
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_snapshot"])
    schema = Features(
        {"input_ids": List(Value("int32")), "loss_mask": List(Value("int8"))}
    )
    iterator = iter(stream)
    rows_written = 0
    token_count = 0
    trained_tokens = 0
    order_digest = hashlib.sha256()
    shards = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while rows_written < recipe["max_prompts"]:
            count = min(1024, recipe["max_prompts"] - rows_written)
            rows = list(islice(iterator, count))
            if len(rows) != count:
                raise ValueError("Dataset ended before the requested training budget.")
            for row in rows:
                order_digest.update(
                    (
                        json.dumps(
                            row["conversations"], sort_keys=True, ensure_ascii=False
                        )
                        + "\n"
                    ).encode()
                )
            name = f"tokens-{len(shards):05d}.arrow"
            partial = output / (name + ".partial")
            writer = ArrowWriter(path=str(partial), features=schema)
            futures = [
                pool.submit(
                    render_conversation,
                    row["conversations"],
                    tokenizer,
                    recipe["max_length"],
                )
                for row in rows
            ]
            for future in futures:
                row = future.result()
                writer.write(row)
                token_count += len(row["input_ids"])
                trained_tokens += sum(row["loss_mask"][1:])
            writer.finalize()
            partial.rename(output / name)
            rows_written += count
            shards.append(name)
            print(
                json.dumps(
                    {
                        "rows": rows_written,
                        "tokens": token_count,
                        "loss_tokens": trained_tokens,
                    }
                ),
                flush=True,
            )
    manifest = {
        **recipe,
        "rows": rows_written,
        "tokens": token_count,
        "loss_tokens": trained_tokens,
        "conversation_order_sha256": order_digest.hexdigest(),
        "datasets_version": datasets.__version__,
        "shards": shards,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
