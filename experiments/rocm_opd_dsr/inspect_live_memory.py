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

"""Inspect and collect CUDA IPC allocations in the stalled job's policy actors."""

import json
from pathlib import Path

import ray
import ray._private.services as services


def collect_memory(worker):
    import gc

    import torch

    before = {
        "free": torch.cuda.mem_get_info()[0],
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    after = {
        "free": torch.cuda.mem_get_info()[0],
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    return {"before": before, "after": after}


if __name__ == "__main__":
    port_file = next(
        Path("/tmp/nrr_49159/ray/session_latest").glob("gcs_server_port_*")
    )
    # The diagnostic container cannot discover the original namespace's raylet PID.
    services.find_node_ids = lambda: [port_file.name.removeprefix("gcs_server_port_")]
    ray.init(
        address="10.32.17.214:" + port_file.read_text().strip(), log_to_driver=False
    )
    actors = [
        x
        for x in ray.util.list_named_actors(all_namespaces=True)
        if x["name"].startswith(("student-", "teacher-"))
    ]
    futures = [
        ray.get_actor(x["name"], namespace=x["namespace"]).__ray_call__.remote(
            collect_memory
        )
        for x in actors
    ]
    report = dict(zip([x["name"] for x in actors], ray.get(futures, timeout=30)))
    Path("experiments/rocm_opd_dsr/run_49159/ipc_memory_probe.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
