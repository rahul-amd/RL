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

"""Native NeMo SFT with historical tokens, masks, ordering and loss reduction."""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.tml_opd_replication.sft_data import HistoricalSFTDataset
from experiments.tml_opd_replication.sft_loss import SummedSFTLoss
from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--verify-restore-audit", type=Path)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    worker = "experiments.tml_opd_replication.audit_worker.AuditedSFTWorker"
    ACTOR_ENVIRONMENT_REGISTRY[worker] = sys.executable
    config.policy["worker_extension_cls_fqn"] = worker
    if config.data["require_complete"]:
        manifest = json.loads(
            (Path(config.data["tokenized_path"]) / "manifest.json").read_text()
        )
        if manifest["rows"] != 384000:
            raise ValueError("Full SFT requires all 384000 historical training rows.")
    dataset = HistoricalSFTDataset(
        config.data["tokenized_path"], config.data["sample_limit"]
    )
    print(config.model_dump(), flush=True)
    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    (
        policy,
        cluster,
        train_loader,
        val_loader,
        _,
        logger,
        checkpointer,
        state,
        config,
    ) = setup(config, tokenizer, dataset, None)
    if args.audit or args.verify_restore_audit:
        before = policy.run_all_workers_single_data("parameter_audit")
        (Path(config.logger["log_dir"]) / "parameter_audit_before.json").write_text(
            json.dumps(before, indent=2) + "\n"
        )
        if args.verify_restore_audit:
            expected = json.loads(args.verify_restore_audit.read_text())["after"]
            if before != expected:
                raise ValueError(
                    "Restored parameters differ from the checkpoint audit."
                )
            print(
                "Checkpoint restore audit passed for every parameter shard.", flush=True
            )
    with checkpointer:
        sft_train(
            policy,
            train_loader,
            val_loader,
            tokenizer,
            SummedSFTLoss(),
            config,
            logger,
            checkpointer,
            state,
        )
    if args.audit:
        after = policy.run_all_workers_single_data("parameter_audit")
        for initial, final in zip(before, after, strict=True):
            assert initial["frozen_sha256"] == final["frozen_sha256"]
            assert initial["adapter_sha256"] != final["adapter_sha256"]
        (Path(config.logger["log_dir"]) / "parameter_audit.json").write_text(
            json.dumps({"before": before, "after": after}, indent=2) + "\n"
        )
        print("Parameter audit passed: base frozen, adapters finite and changed.")


if __name__ == "__main__":
    main()
