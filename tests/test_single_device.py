import copy

import pytest

torch = pytest.importorskip("torch")

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory, AgentTurn
from mapd.trainer.opsd import PrivilegedInformation
from mapd.trainer.single_device import (
    load_smoke_checkpoint,
    optimize_replay_group,
    save_smoke_checkpoint,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        privileged = "PRIVATE" in messages[1]["content"]
        return [1, 9 if privileged else 2]

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [3, 4]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.output = torch.nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return type("Output", (), {"logits": self.output(self.embedding(input_ids))})


def _trajectory(reward, response_ids):
    return AgentTrajectory(
        example_id="q1",
        prompt="Where?",
        reward=reward,
        terminated=True,
        final_answer="Paris" if reward else "London",
        turns=[
            AgentTurn(
                turn_index=1,
                model_output="<answer>x</answer>",
                action="answer",
                token_ids=response_ids,
            )
        ],
    )


def test_single_device_step_updates_and_reloads_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sample = QASample(id="q1", question="Where?", answers=["Paris"])
    trajectories = [_trajectory(1.0, [3, 4]), _trajectory(0.0, [5, 6])]

    result = optimize_replay_group(
        model,
        TinyTokenizer(),
        optimizer,
        sample,
        trajectories,
        PrivilegedInformation("protocol", "PRIVATE"),
        max_sequence_length=16,
    )

    assert result.response_tokens == 4
    assert result.loss.distillation_tokens == 4
    assert result.grad_norm > 0
    assert result.sampled_parameter_delta > 0
    checkpoint = save_smoke_checkpoint(
        tmp_path,
        model,
        optimizer,
        result,
        base_model="tiny",
        step=1,
    )
    expected = model.embedding.weight.detach().clone()
    with torch.no_grad():
        model.embedding.weight.zero_()
    assert load_smoke_checkpoint(checkpoint, model, optimizer) == 1
    assert torch.equal(model.embedding.weight, expected)


def test_single_device_step_applies_frozen_reference_policy_kl():
    torch.manual_seed(11)
    model = TinyModel()
    reference_model = copy.deepcopy(model)
    with torch.no_grad():
        reference_model.output.weight[0].add_(2.0)
    reference_model.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sample = QASample(id="q1", question="Where?", answers=["Paris"])
    trajectories = [_trajectory(1.0, [3, 4]), _trajectory(0.0, [5, 6])]

    result = optimize_replay_group(
        model,
        TinyTokenizer(),
        optimizer,
        sample,
        trajectories,
        None,
        reference_model=reference_model,
        beta=0.1,
        max_sequence_length=16,
    )

    assert result.loss.reference_kl > 0
    assert all(parameter.grad is None for parameter in reference_model.parameters())
