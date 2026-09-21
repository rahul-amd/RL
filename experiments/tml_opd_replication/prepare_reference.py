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

"""Fetch the pinned historical renderer used by the recipe parity test."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    relative = "reference_330b73d/tinker_cookbook/renderers.py"
    upstream_path = relative.split("/", 1)[1]
    url = (
        "https://raw.githubusercontent.com/thinking-machines-lab/"
        f"tinker-cookbook/{manifest['reference_commit']}/{upstream_path}"
    )
    with urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != manifest["source_sha256"][relative]:
        raise ValueError("Historical renderer checksum mismatch")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(destination)


if __name__ == "__main__":
    main()
