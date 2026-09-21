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

"""Export the checksummed Prime-RL DSR splits; source data stays local."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("data_manifest.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(__file__).resolve().parent
    for split, metadata in manifest["splits"].items():
        source = Path(metadata["source"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {source}")
        rows = pq.read_table(source, columns=["question", "answer"]).to_pylist()
        if len(rows) != metadata["rows"]:
            raise ValueError(f"Unexpected row count for {split}: {len(rows)}")
        payload = "".join(
            json.dumps(
                {
                    "input": manifest["instruction"] + row["question"],
                    "output": row["answer"],
                    "source_row": i,
                },
                ensure_ascii=False,
            )
            + "\n"
            for i, row in enumerate(rows)
        ).encode()
        if hashlib.sha256(payload).hexdigest() != metadata["export_sha256"]:
            raise ValueError(f"Export checksum mismatch: {split}")
        (root / f"{split}.jsonl").write_bytes(payload)
        print(f"Exported {len(rows)} {split} rows")


if __name__ == "__main__":
    main()
