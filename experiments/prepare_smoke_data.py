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

"""Generate the deterministic arithmetic inputs for the ROCm smoke recipes."""

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    for split, indices in [("train", range(64)), ("validation", range(64, 72))]:
        rows = [
            {
                "input": f"Add {i} and 3. Give the sum in a short sentence.",
                "output": f"The sum of {i} and 3 is {i + 3}.",
            }
            for i in indices
        ]
        (root / "rocm_sft" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    rows = []
    for i in range(32):
        a, b, c = 37 + 3 * i, 19 + 2 * i, 7 + i
        rows.append(
            {
                "input": f"Calculate {a} * {b} + {c}. Show a short calculation and put the final integer inside "
                + "\\boxed{}.",
                "output": str(a * b + c),
            }
        )
    (root / "rocm_rl/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
