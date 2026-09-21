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

"""Exercise the native ROCm refit group with real multi-GPU broadcasts."""

import os

import torch

from nemo_rl.distributed.refit_watchdog import RefitAborted
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    group = StatelessProcessGroup("127.0.0.1", 1998, rank, world_size)
    group.init_nccl_communicator(rank)
    stream = torch.cuda.Stream()
    for step in range(8):
        source = step % world_size
        dtype = torch.bfloat16 if step % 2 else torch.float32
        tensor = torch.full((1024 * 1024,), float(rank), dtype=dtype, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        group.broadcast(tensor, source, stream=stream)
        stream.synchronize()
        assert torch.all(tensor == source), (rank, step)
    group.abort()
    group.abort()
    try:
        group.broadcast(tensor, 0)
    except RefitAborted:
        pass
    else:
        raise AssertionError("An aborted group accepted a broadcast")
    print(f"Rank {rank}: eight broadcasts and abort checks passed", flush=True)


if __name__ == "__main__":
    main()
