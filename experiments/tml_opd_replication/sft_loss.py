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

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn


class SummedSFTLoss(NLLLossFn):
    """Tinker's summed token gradients, with mean NLL retained for logging."""

    def __call__(self, next_token_logprobs, data, global_valid_seqs, global_valid_toks):
        loss, metrics = super().__call__(
            next_token_logprobs, data, global_valid_seqs, global_valid_toks
        )
        return loss * global_valid_toks, metrics
