import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import (
    DistillationLossConfig,
    SampledReverseKLLossFn,
)


def test_sampled_objective_expected_gradient_matches_reverse_kl():
    logits = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.tensor([0.8, -0.1, 0.0], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    logq = teacher_logits.log_softmax(-1)
    sampled = torch.cat([torch.zeros(1), logp.detach()]).unsqueeze(0).requires_grad_()
    teacher = torch.cat([torch.zeros(1), logq]).unsqueeze(0).requires_grad_()
    data = {
        "generation_logprobs": sampled,
        "teacher_logprobs": teacher,
        "token_mask": torch.cat([torch.zeros(1), logp.detach().exp()]).unsqueeze(0),
        "sample_mask": torch.ones(1),
    }
    loss_fn = SampledReverseKLLossFn(
        DistillationLossConfig(sampled_token_reduction="sum")
    )
    loss, _ = loss_fn(logp.unsqueeze(0), data, torch.tensor(1), torch.tensor(1.0))
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(
        (logp.exp() * (logp - logq)).sum(), logits, retain_graph=True
    )[0]
    torch.testing.assert_close(actual, expected)
    loss.backward()
    assert sampled.grad is None and teacher.grad is None


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_sampled_objective_masks_padding_and_accumulates_microbatches(reduction):
    current = torch.tensor(
        [[-1.2, -0.4, float("nan")], [-0.2, -0.8, -0.5]], requires_grad=True
    )
    sampled = torch.tensor([[0.0, -1.0, -0.6, float("-inf")], [0.0, -0.4, -0.9, -0.3]])
    teacher = torch.tensor([[0.0, -0.8, -0.9, float("-inf")], [0.0, -0.7, -0.5, -0.8]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1]])
    data = {
        "generation_logprobs": sampled,
        "teacher_logprobs": teacher,
        "token_mask": mask,
        "sample_mask": torch.ones(2),
    }
    loss_fn = SampledReverseKLLossFn(
        DistillationLossConfig(sampled_token_reduction=reduction)
    )
    loss, _ = loss_fn(current, data, torch.tensor(2), torch.tensor(5))
    loss.backward()
    first_gradient = current.grad.clone()
    current.grad.zero_()
    accumulated = 0
    for i in range(2):
        mb = {k: v[i : i + 1] for k, v in data.items()}
        part, _ = loss_fn(current[i : i + 1], mb, torch.tensor(2), torch.tensor(5))
        part.backward()
        accumulated += part.detach()
    torch.testing.assert_close(current.grad, first_gradient)
    torch.testing.assert_close(accumulated, loss.detach())
    assert current.grad[0, 2] == 0
    valid = mask[:, 1:].bool()
    expected = -torch.exp(current.detach()[valid] - sampled[:, 1:][valid]) * (
        teacher[:, 1:][valid] - sampled[:, 1:][valid]
    )
    if reduction == "mean":
        expected /= 5
    torch.testing.assert_close(current.grad[valid], expected)


def test_rollout_logprobs_align_after_prompt_and_batch_padding():
    from nemo_rl.algorithms.distillation import prepare_distillation_messages
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    messages = [
        [
            {"role": "user", "token_ids": torch.tensor([1, 2, 3])},
            {
                "role": "assistant",
                "token_ids": torch.tensor([4, 5]),
                "generation_logprobs": torch.tensor([-0.1, -0.2]),
            },
        ],
        [
            {"role": "user", "token_ids": torch.tensor([6])},
            {
                "role": "assistant",
                "token_ids": torch.tensor([7, 8]),
                "generation_logprobs": torch.tensor([-0.3, -0.4]),
            },
        ],
    ]
    prepare_distillation_messages(messages, include_sampling_logprobs=True)
    flat, lengths = batched_message_log_to_flat_message(
        messages, pad_value_dict={"token_ids": 0}
    )
    torch.testing.assert_close(
        flat["generation_logprobs"],
        torch.tensor([[0, 0, 0, -0.1, -0.2], [0, -0.3, -0.4, 0, 0]]),
    )
    torch.testing.assert_close(
        flat["token_loss_mask"], torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]])
    )
    torch.testing.assert_close(lengths, torch.tensor([5, 3], dtype=torch.int32))
