# On-policy Distillation

We provide an example on-policy distillation experiment using the [DeepScaler dataset](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview).

> [!NOTE]
> Distillation currently supports the DTensor and vLLM generation backend. Megatron generation/training paths are not supported yet.

## On-policy Distillation Single Node

To run on-policy distillation on a single GPU using `Qwen/Qwen3-1.7B-Base` as the student and `Qwen/Qwen3-4B` as the teacher:

```sh
uv run python examples/run_distillation.py
```

Customize parameters with command-line overrides. For example:

```sh
uv run python examples/run_distillation.py \
  policy.model_name="Qwen/Qwen3-1.7B-Base" \
  teacher.model_name="Qwen/Qwen3-4B" \
  cluster.gpus_per_node=8
```

### On-policy Distillation with NeMo Gym

On-policy distillation can use NeMo Gym for multi-step or multi-turn rollout collection. In this mode, NeMo RL exposes the student vLLM generation worker as an OpenAI-compatible HTTP server, NeMo Gym runs the environment interaction, and the resulting student samples are used for teacher-logit distillation.

Use the NeMo Gym distillation entrypoint with the example config. The checked-in config uses placeholder dataset paths, so override them for your local data:

```sh
uv run python examples/nemo_gym/run_distillation_nemo_gym.py \
  --config examples/nemo_gym/distillation_qwen3_0_6b.yaml \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/validation.jsonl
```

The config must enable the vLLM async HTTP server and NeMo Gym:

```yaml
policy:
  generation:
    backend: vllm
    vllm_cfg:
      async_engine: true
      expose_http_server: true

env:
  should_use_nemo_gym: true
```

NeMo Gym controls the rollout turn count from its environment and agent configuration. The standard distillation `distillation.max_rollout_turns` setting is not used by the NeMo Gym rollout path.

## On-policy Distillation Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_distillation.py --config examples/configs/distillation_math.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/distill_2nodes' logger.wandb_enabled=True logger.wandb.name='distill-2nodes'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

## Sampled-token reverse KL

Set `loss_fn.kl_type=sampled_reverse` in the standard distillation entrypoint to
score each student-generated token with the frozen teacher's full-vocabulary
log-probability. This mode does not use `distillation.topk_logits_k`.

The detached per-token advantage is `teacher_logprob - rollout_logprob`; the loss
is `-exp(current_logprob - rollout_logprob) * advantage`, masked to response
tokens. `loss_fn.sampled_token_reduction=mean` normalizes gradients by the global
valid-token count; `sum` preserves summed token gradients. Reported loss metrics
remain token means in both modes. Choose the reduction together with the learning
rate because it changes gradient scale. There is no importance-ratio clipping.

For synchronous on-policy training, use one optimizer update per fresh rollout
batch and unfiltered temperature-1 sampling. The historical rank-128 LoRA
SFT/OPD recipe, AMD ROCm launchers, and explicit evaluation assumptions are in
[`experiments/tml_opd_replication`](../../../experiments/tml_opd_replication/README.md).
