from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutLossInput:
    reward: float
    old_action_log_probs: list[float]
    new_action_log_probs: list[float]
    reference_token_kls: list[float]
    student_log_distributions: list[list[float]]
    privileged_log_distributions: list[list[float]] | None
    action_mask: list[bool]

    def __post_init__(self) -> None:
        lengths = {
            len(self.old_action_log_probs),
            len(self.new_action_log_probs),
            len(self.reference_token_kls),
            len(self.student_log_distributions),
            len(self.action_mask),
        }
        if len(lengths) != 1:
            raise ValueError("all per-token rollout fields must have identical lengths")
        if self.privileged_log_distributions is not None and len(self.privileged_log_distributions) != len(self.action_mask):
            raise ValueError("privileged distributions must align with response positions")


def group_relative_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    if not rewards:
        raise ValueError("a GRPO group cannot be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    return [0.0 for _ in rewards] if std < epsilon else [(reward - mean) / (std + epsilon) for reward in rewards]


def grpo_loss(
    group: list[RolloutLossInput],
    advantages: list[float],
    *,
    clip_low: float,
    clip_high: float,
    beta: float,
) -> float:
    trajectory_objectives = []
    for rollout, advantage in zip(group, advantages):
        values = []
        for old, new, ref_kl, active in zip(
            rollout.old_action_log_probs,
            rollout.new_action_log_probs,
            rollout.reference_token_kls,
            rollout.action_mask,
        ):
            if active:
                ratio = math.exp(new - old)
                clipped = min(max(ratio, 1.0 - clip_low), 1.0 + clip_high)
                values.append(min(ratio * advantage, clipped * advantage) - beta * ref_kl)
        trajectory_objectives.append(sum(values) / len(values) if values else 0.0)
    return -sum(trajectory_objectives) / len(trajectory_objectives)


def torch_grpo_loss(
    new_action_log_probs,
    old_action_log_probs,
    advantages,
    reference_token_kls,
    action_mask,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    beta: float = 0.0,
):
    import torch

    ratios = torch.exp(new_action_log_probs - old_action_log_probs)
    advantage = advantages.unsqueeze(-1)
    objective = torch.minimum(
        ratios * advantage,
        ratios.clamp(1.0 - clip_low, 1.0 + clip_high) * advantage,
    ) - beta * reference_token_kls
    mask = action_mask.to(objective.dtype)
    return -((objective * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).mean()

