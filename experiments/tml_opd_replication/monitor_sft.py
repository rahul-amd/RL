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

import argparse
import fcntl
import getpass
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
RECOVERABLE = {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "TIMEOUT"}


def command(args):
    return subprocess.run(
        args, cwd=REPO, check=True, text=True, capture_output=True, timeout=30
    ).stdout


def latest_checkpoint(root):
    checkpoints = sorted(
        (p for p in root.glob("step_*") if re.fullmatch(r"step_\d+", p.name)),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No completed checkpoint in {root}")
    latest = checkpoints[-1]
    for name in (
        "config.yaml",
        "training_info.json",
        "train_dataloader.pt",
        "policy/weights/model/adapter_model.safetensors",
        "policy/weights/model/adapter_config.json",
        "policy/optimizer/optim/.metadata",
    ):
        if not (latest / name).is_file():
            raise FileNotFoundError(latest / name)
    info = json.loads((latest / "training_info.json").read_text())
    if info["total_steps"] != int(latest.name.split("_")[1]):
        raise ValueError(f"Checkpoint step mismatch: {latest}")
    return latest, info["total_steps"]


def completed_step(log):
    matches = re.findall(r"Step (\d+)/\d+.*?Total step time: ([\d.]+)s", log, re.DOTALL)
    return int(matches[-1][0]) if matches else None


def infrastructure_failure(log):
    if any(
        term in log
        for term in ("OutOfMemoryError", "out of memory", "Nonfinite", "nan gradient")
    ):
        return False
    return any(
        term in log
        for term in (
            "ActorDiedError",
            "NodeDiedError",
            "collective operation timeout",
            "Checkpoint background process failed",
            "Connection reset by peer",
            "Invalid generic resource (gres) specification",
        )
    )


def job_states(jobs):
    output = command(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(map(str, jobs)),
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,User%80,JobName%80",
        ]
    )
    result = {}
    for row in output.splitlines():
        fields = row.split("|")
        job, status, user, name = (field.strip() for field in fields[:4])
        if int(job) not in jobs:
            continue
        if user != getpass.getuser() or not name.startswith("tml-sft-"):
            raise ValueError(f"Refusing to manage unexpected job {job}, {user}, {name}")
        result[int(job)] = status.split()[0].rstrip("+")
    return result


def persist(path, state):
    state["heartbeat_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def event(state, message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)
    state["last_event"] = message


def submit(state, config, gpus):
    checkpoint, step = latest_checkpoint(
        EXPERIMENT / f"run_{config['run_id']}" / "checkpoints"
    )
    script = "sft_multinode.sbatch" if gpus == 32 else "sft.sbatch"
    export = f"ALL,TML_SFT_RUN_ID={config['run_id']},TML_SFT_CONFIG={checkpoint / 'config.yaml'}"
    state["status"] = "submitting"
    persist(config["state_path"], state)
    # An uncertain submission must not be retried blindly: it may already exist.
    try:
        output = command(
            [
                "sbatch",
                f"--export={export}",
                f"--job-name=tml-sft-watch-{gpus}gpu",
                str(EXPERIMENT / script),
                "sft.max_num_steps=3000",
                f"checkpointing.save_period={config['save_period']}",
                f"policy.dtensor_cfg.dp_replicate_size={4 if gpus == 32 else 1}",
            ]
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        state["status"] = "needs_attention"
        event(
            state,
            "Submission failed or uncertain; inspect Slurm before restarting monitor.",
        )
        persist(config["state_path"], state)
        raise
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise RuntimeError(f"Unrecognized sbatch response: {output}")
    job = int(match[1])
    event(state, f"Submitted {job} with {gpus} GPUs from completed step {step}")
    state["status"] = "watching"
    return job


def tick(state, config):
    now = time.time()
    run = EXPERIMENT / f"run_{config['run_id']}"
    _, checkpoint_step = latest_checkpoint(run / "checkpoints")
    if checkpoint_step > state["checkpoint_step"]:
        state["restarts_without_progress"] = 0
        state["checkpoint_step"] = checkpoint_step
    jobs = [state["active_job"]]
    if state["upgrade_job"]:
        jobs.append(state["upgrade_job"])
    try:
        statuses = job_states(jobs)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        event(state, f"Slurm query failed; leaving jobs untouched: {error}")
        return True
    active = state["active_job"]
    status = statuses.get(active, "UNKNOWN")
    upgrade = state["upgrade_job"]
    upgrade_status = statuses.get(upgrade, "UNKNOWN")
    state["slurm_states"] = statuses
    if upgrade and upgrade_status == "RUNNING" and status != "UNKNOWN":
        upgrade_log = EXPERIMENT / f"sft-{upgrade}.log"
        if upgrade_log.exists() and "Waiting for run" in upgrade_log.read_text():
            if status in ACTIVE:
                command(["scancel", str(active)])
            event(
                state,
                f"Promoting {upgrade} to 32 GPUs; superseded job {active}, checkpoint {checkpoint_step}",
            )
            state.update(
                active_job=upgrade,
                active_gpus=32,
                upgrade_job=None,
                progress_at=now,
                running_since=None,
                completed_step=None,
            )
            return True
    elif upgrade and upgrade_status not in ACTIVE | {"UNKNOWN"}:
        event(
            state,
            f"Upgrade job {upgrade} ended {upgrade_status}; current training continues",
        )
        state["upgrade_job"] = None
        state["upgrade_attention"] = True
    log_path = EXPERIMENT / f"sft-{active}.log"
    log = log_path.read_text() if log_path.exists() else ""
    progress = completed_step(log)
    if progress is not None and progress != state["completed_step"]:
        state["completed_step"] = progress
        state["progress_at"] = now
    if status == "RUNNING":
        if state["running_since"] is None:
            state["running_since"] = now
            state["progress_at"] = now
        limit = (
            config["startup_seconds"] if progress is None else config["stall_seconds"]
        )
        explicit_failure = (
            "Checkpoint background process failed during initialization" in log
        )
        if explicit_failure or now - state["progress_at"] > limit:
            event(
                state,
                f"Stopping stalled/failed job {active}; last completed update {progress}",
            )
            state["restart_requested"] = True
            persist(config["state_path"], state)
            command(["scancel", str(active)])
    elif status not in ACTIVE | {"UNKNOWN"}:
        if status == "COMPLETED" and checkpoint_step >= config["target_steps"]:
            if upgrade and upgrade_status in ACTIVE:
                command(["scancel", str(upgrade)])
            state["status"] = "complete"
            event(state, "SFT complete with final checkpoint")
            return False
        recoverable = (
            status in RECOVERABLE
            or state["restart_requested"]
            or (status == "FAILED" and infrastructure_failure(log))
        )
        if (
            not recoverable
            or state["restarts_without_progress"]
            >= config["max_restarts_without_progress"]
        ):
            if upgrade and upgrade_status in ACTIVE:
                command(["scancel", str(upgrade)])
            state["status"] = "needs_attention"
            event(state, f"Job {active} ended {status}; automatic recovery stopped")
            return False
        state["restarts_without_progress"] += 1
        state["active_job"] = submit(state, config, state["active_gpus"])
        state.update(
            restart_requested=False,
            running_since=None,
            completed_step=None,
            progress_at=now,
        )
    event(
        state,
        f"Job {state['active_job']}: {status}; update={progress}; checkpoint={checkpoint_step}; upgrade={upgrade}:{upgrade_status}",
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["state_path"] = Path(config["state_path"])
    with config["state_path"].with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(config["state_path"].read_text())
        if state["status"] not in {"watching", "starting"}:
            raise RuntimeError(f"Monitor requires review: {state['status']}")
        while True:
            keep_running = tick(state, config)
            persist(config["state_path"], state)
            if not keep_running or args.once:
                break
            time.sleep(config["poll_seconds"])


if __name__ == "__main__":
    main()
