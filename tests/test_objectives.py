import math

import pytest

from mapd.trainer.grpo import RolloutLossInput, group_relative_advantages
from mapd.trainer.mapd_trainer import compute_mapd_loss
from mapd.trainer.opsd import reverse_kl


def _rollout(reward: float, privileged: bool = True) -> RolloutLossInput:
    return RolloutLossInput(
        reward=reward,
        old_action_log_probs=[math.log(0.5)],
        new_action_log_probs=[math.log(0.5)],
        reference_token_kls=[0.0],
        student_log_distributions=[[math.log(0.5), math.log(0.5)]],
        privileged_log_distributions=[[math.log(0.75), math.log(0.25)]] if privileged else None,
        action_mask=[True],
    )


def test_group_advantages_and_joint_loss_match_equations():
    assert group_relative_advantages([1.0, 0.0]) == pytest.approx([1.0, -1.0])
    expected_kl = reverse_kl(
        [math.log(0.5), math.log(0.5)], [math.log(0.75), math.log(0.25)]
    )
    result = compute_mapd_loss([_rollout(1.0), _rollout(0.0)], lambda_opsd=0.05)
    assert result.grpo == pytest.approx(0.0)
    assert result.reference_kl == pytest.approx(0.0)
    assert result.opsd == pytest.approx(expected_kl)
    assert result.total == pytest.approx(0.05 * expected_kl)
    assert result.distillation_tokens == 2


def test_grpo_includes_the_paper_reference_policy_kl_penalty():
    positive = _rollout(1.0)
    negative = _rollout(0.0)
    positive = RolloutLossInput(
        **{**positive.__dict__, "reference_token_kls": [0.4]}
    )
    negative = RolloutLossInput(
        **{**negative.__dict__, "reference_token_kls": [0.2]}
    )

    result = compute_mapd_loss(
        [positive, negative], beta=0.1, lambda_opsd=0.0
    )

    assert result.reference_kl == pytest.approx(0.3)
    assert result.grpo == pytest.approx(0.03)
    assert result.total == pytest.approx(0.03)
