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
"""Wait for the allocated Ray nodes, then run native SFT."""

import os
import time

import ray

from experiments.tml_opd_replication.run_sft import main

if __name__ == "__main__":
    expected_nodes = int(os.environ["TML_NODE_COUNT"])
    expected_gpus = int(os.environ["TML_GPU_COUNT"])
    ray.init(address=os.environ["RAY_ADDRESS"])
    for attempt in range(120):
        nodes = [node for node in ray.nodes() if node["Alive"]]
        if len(nodes) == expected_nodes:
            if any(node["Resources"].get("GPU", 0) != expected_gpus for node in nodes):
                raise RuntimeError(f"Unexpected GPU allocation: {nodes}")
            print(
                f"Ray ready: {expected_nodes} nodes, "
                f"{expected_nodes * expected_gpus} GPUs",
                flush=True,
            )
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Expected {expected_nodes} Ray nodes, found {len(nodes)}")
    ray.shutdown()
    main()
