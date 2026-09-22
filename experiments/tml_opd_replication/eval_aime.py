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

"""Evaluate historical SFT/OPD adapters and the Qwen3 teacher on AIME24."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = Path("/shared_silo/scratch/hf_cache/hub")
BASE = (
    CACHE
    / "models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
)
TEACHER = (
    CACHE / "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
ADAPTER = ROOT / "run_49416/checkpoints/step_3000/policy/weights/model"
MERGED = ROOT / "checkpoints/inference_sft_3000"
DATASET_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
DATA = (
    CACHE
    / f"datasets--HuggingFaceH4--aime_2024/snapshots/{DATASET_REVISION}/data/train-00000-of-00001.parquet"
)
PROTOCOL = {
    "dataset": "HuggingFaceH4/aime_2024",
    "dataset_revision": DATASET_REVISION,
    "num_problems": 30,
    "repeats": 32,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_total_tokens": 32768,
    "seed_base": 1234,
    "instruction": "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "renderer": "qwen3_historical.jinja",
    "metric": "mean sample accuracy (pass@1)",
    "scoring": "math-verify on last complete boxed expression; missing box is incorrect",
    "historical_deviation": "Published historical AIME24 sampling settings were not specified.",
    "inference_dtype": "bfloat16",
    "adapter": str(ADAPTER),
    "base": str(BASE),
    "teacher": str(TEACHER),
}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def last_box(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def merge():
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (MERGED / "merge_validation.json").exists():
        print("Using validated merged export", MERGED, flush=True)
        return
    torch.set_float32_matmul_precision("highest")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float32, device_map="cuda:0", attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    saved = load_file(str(ADAPTER / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    assert set(saved) == set(loaded), (
        set(saved) - set(loaded),
        set(loaded) - set(saved),
    )
    assert len(saved) == 506
    for key, value in saved.items():
        assert torch.equal(value, loaded[key].cpu()), key
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.chat_template = (ROOT / "qwen3_historical.jinja").read_text()
    prompts = [
        "Compute 17 times 23.",
        "Solve x squared minus five x plus six equals zero.",
    ]
    batches = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        for p in prompts
    ]
    with torch.inference_mode():
        before = [model(**batch).logits.cpu() for batch in batches]
        merged = model.merge_and_unload(safe_merge=True).eval()
        after = [merged(**batch).logits.cpu() for batch in batches]
    errors = [(a - b).abs().max().item() for a, b in zip(before, after)]
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=0.01, rtol=0.001)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.to(torch.bfloat16).save_pretrained(MERGED, max_shard_size="4GB")
    tokenizer.save_pretrained(MERGED)
    report = {
        "adapter_tensors_verified_exactly": len(saved),
        "modules": len(saved) // 2,
        "includes_lm_head": any("lm_head" in k for k in saved),
        "fp32_logit_max_absolute_errors": errors,
        "export_dtype": "bfloat16",
        "adapter_sha256": hashlib.file_digest(
            (ADAPTER / "adapter_model.safetensors").open("rb"), "sha256"
        ).hexdigest(),
    }
    write_json(MERGED / "merge_validation.json", report)
    print("MERGE_VALIDATED", json.dumps(report), flush=True)


def worker(args):
    import pyarrow.parquet as pq
    from math_verify import parse, verify
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import nemo_rl  # noqa: F401

    rows = pq.read_table(DATA).to_pylist()
    assert len(rows) == 30 and len({r["problem"] for r in rows}) == 30
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    tokenizer.chat_template = (ROOT / "qwen3_historical.jinja").read_text()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["problem"] + PROTOCOL["instruction"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        for r in rows
    ]
    assert all(
        isinstance(p, list) and p and all(isinstance(t, int) for t in p)
        for p in prompts
    )
    print(
        "PROMPTS_VALIDATED",
        len(prompts),
        min(map(len, prompts)),
        max(map(len, prompts)),
        flush=True,
    )
    llm = LLM(
        model=str(TEACHER if args.model == "teacher" else MERGED),
        tokenizer=str(BASE),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=PROTOCOL["max_total_tokens"],
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=True,
        generation_config="vllm",
        seed=PROTOCOL["seed_base"],
    )
    output_path = args.output / f"{args.model}-rank{args.rank}.jsonl"
    done = set()
    if output_path.exists():
        done = {json.loads(line)["id"] for line in output_path.read_text().splitlines()}
    with output_path.open("a", buffering=1) as stream:
        for repeat in range(args.rank, PROTOCOL["repeats"], args.workers):
            indices = [i for i in range(30) if f"{repeat}:{i}" not in done]
            if not indices:
                continue
            params = [
                SamplingParams(
                    temperature=PROTOCOL["temperature"],
                    top_p=PROTOCOL["top_p"],
                    top_k=PROTOCOL["top_k"],
                    min_p=0.0,
                    max_tokens=PROTOCOL["max_total_tokens"] - len(prompts[i]),
                    seed=PROTOCOL["seed_base"] + repeat * 30 + i,
                    stop_token_ids=[151643, 151645],
                )
                for i in indices
            ]
            generated = llm.generate(
                [{"prompt_token_ids": prompts[i]} for i in indices],
                params,
                use_tqdm=False,
            )
            for i, result in zip(indices, generated, strict=True):
                out = result.outputs[0]
                boxed = last_box(out.text)
                correct = boxed is not None and bool(
                    verify(parse(str(rows[i]["answer"])), parse("$" + boxed + "$"))
                )
                record = {
                    "id": f"{repeat}:{i}",
                    "problem_index": i,
                    "repeat": repeat,
                    "seed": PROTOCOL["seed_base"] + repeat * 30 + i,
                    "answer": str(rows[i]["answer"]),
                    "boxed": boxed,
                    "correct": correct,
                    "text": out.text,
                    "prompt_tokens": len(prompts[i]),
                    "generated_tokens": len(out.token_ids),
                    "finish_reason": out.finish_reason,
                    "stop_reason": out.stop_reason,
                }
                stream.write(json.dumps(record) + "\n")
            print(
                f"PROGRESS model={args.model} rank={args.rank} repeat={repeat} samples={len(indices)}",
                flush=True,
            )


def summarize(args):
    import numpy as np
    import wandb

    records = [
        json.loads(line)
        for p in args.output.glob(f"{args.model}-rank*.jsonl")
        for line in p.read_text().splitlines()
    ]
    expected = {f"{r}:{i}" for r in range(32) for i in range(30)}
    assert len(records) == len(expected) and {r["id"] for r in records} == expected
    scores = np.array(
        [
            [
                next(x["correct"] for x in records if x["id"] == f"{r}:{i}")
                for r in range(32)
            ]
            for i in range(30)
        ],
        dtype=float,
    )
    problem_scores = scores.mean(axis=1)
    rng = np.random.default_rng(1234)
    ci = np.quantile(
        problem_scores[rng.integers(0, 30, (10000, 30))].mean(axis=1), [0.025, 0.975]
    )
    result = {
        "model": args.model,
        "samples": len(records),
        "accuracy": float(scores.mean()),
        "problem_bootstrap_95_ci": ci.tolist(),
        "per_problem_accuracy": problem_scores.tolist(),
        "per_repeat_accuracy": scores.mean(axis=0).tolist(),
        "truncated_fraction": sum(r["finish_reason"] == "length" for r in records)
        / len(records),
        "missing_box_fraction": sum(r["boxed"] is None for r in records) / len(records),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in records)
        / len(records),
    }
    if args.model == "opd":
        baseline_path = ROOT / "run_aime_49575/sft-summary.json"
        baseline = json.loads(baseline_path.read_text())
        result["initial_sft_accuracy"] = baseline["accuracy"]
        result["accuracy_delta_from_sft"] = result["accuracy"] - baseline["accuracy"]
        result["adapter"] = str(ADAPTER)
    write_json(args.output / f"{args.model}-summary.json", result)
    run = wandb.init(
        entity="rahular",
        project="opd-tests",
        job_type="evaluation",
        id=f"tml-aime24-{args.model}-{args.output.name}",
        resume="allow",
        config=PROTOCOL,
        settings=wandb.Settings(disable_code=True),
    )
    run.log({k: v for k, v in result.items() if isinstance(v, (int, float))})
    run.summary.update(result)
    run.finish()
    print("EVALUATION_COMPLETE", json.dumps(result), flush=True)


def watch(args):
    job_id = args.output.name.removeprefix("run_aime_")
    assert job_id.isdigit(), args.output
    while True:
        accounting = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--noheader",
                "--parsable2",
                "-o",
                "JobID,State,ExitCode",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        states = {
            line.split("|")[0]: line.split("|")[1]
            for line in accounting.splitlines()
            if "." not in line.split("|")[0]
        }
        counts = {
            model: sum(
                sum(1 for _ in p.open())
                for p in args.output.glob(f"{model}-rank*.jsonl")
            )
            for model in ("sft", "teacher")
        }
        expected = [f"{job_id}_{i}" for i in range(2)]
        failures = {
            key: value
            for key, value in states.items()
            if value.split()[0]
            in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
            }
        }
        complete = all(states.get(key) == "COMPLETED" for key in expected)
        status = (
            "needs_attention" if failures else "complete" if complete else "running"
        )
        report = {
            "time": time.time(),
            "state": status,
            "jobs": states,
            "samples": counts,
        }
        write_json(args.output / "watch_state.json", report)
        print(json.dumps(report), flush=True)
        if failures:
            raise RuntimeError(f"Evaluation jobs require attention: {failures}")
        if complete:
            summaries = {
                model: json.loads((args.output / f"{model}-summary.json").read_text())
                for model in ("sft", "teacher")
            }
            comparison = {
                "summaries": summaries,
                "teacher_minus_sft_accuracy": summaries["teacher"]["accuracy"]
                - summaries["sft"]["accuracy"],
            }
            write_json(args.output / "comparison.json", comparison)
            print(json.dumps(comparison), flush=True)
            return
        time.sleep(30)


def main():
    global ADAPTER, MERGED
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["merge", "worker", "run", "summary", "watch"], default="run"
    )
    parser.add_argument("--model", choices=["sft", "teacher", "opd"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.adapter is not None:
        ADAPTER = args.adapter.resolve()
        MERGED = (
            ROOT
            / "checkpoints"
            / f"inference_{ADAPTER.parents[4].name}_{ADAPTER.parents[2].name}"
        )
        PROTOCOL["adapter"] = str(ADAPTER)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "watch":
        watch(args)
    elif args.mode == "merge":
        merge()
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "summary":
        summarize(args)
    else:
        write_json(args.output / f"{args.model}-protocol.json", PROTOCOL)
        import torch

        assert torch.cuda.device_count() == args.workers, torch.cuda.device_count()
        visible_devices = os.environ.get(
            "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.workers)))
        ).split(",")
        assert len(visible_devices) == args.workers, visible_devices
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--output",
            str(args.output),
            "--workers",
            str(args.workers),
        ]
        if args.adapter is not None:
            command += ["--adapter", str(args.adapter)]
        if args.model != "teacher":
            subprocess.run(command + ["--mode", "merge"], check=True)
        processes, logs = [], []
        try:
            for rank in range(args.workers):
                log = (args.output / f"{args.model}-rank{rank}.log").open("a")
                logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_devices[rank])
                env["TRITON_CACHE_DIR"] = f"{os.environ['TRITON_CACHE_DIR']}/rank{rank}"
                env.pop("ROCR_VISIBLE_DEVICES", None)
                env.pop("HIP_VISIBLE_DEVICES", None)
                processes.append(
                    subprocess.Popen(
                        command + ["--mode", "worker", "--rank", str(rank)],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            while any(p.poll() is None for p in processes):
                failed = [
                    (i, p.returncode)
                    for i, p in enumerate(processes)
                    if p.poll() not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"Generation workers failed: {failed}")
                counts = {
                    p.name: sum(1 for _ in p.open())
                    for p in args.output.glob(f"{args.model}-rank*.jsonl")
                }
                write_json(
                    args.output / f"{args.model}-status.json",
                    {"time": time.time(), "samples": counts, "state": "running"},
                )
                time.sleep(30)
            assert all(p.returncode == 0 for p in processes)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
            for log in logs:
                log.close()
        summarize(args)
        write_json(
            args.output / f"{args.model}-status.json",
            {"time": time.time(), "state": "complete"},
        )


if __name__ == "__main__":
    main()
