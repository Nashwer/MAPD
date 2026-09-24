from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory
from mapd.mas.schema import SynthesisArtifact
from mapd.trainer.grpo import RolloutLossInput, grpo_loss, group_relative_advantages
from mapd.trainer.opsd import PrivilegedInformation, reverse_kl, select_privileged_information


@dataclass(frozen=True)
class LossBreakdown:
    grpo: float
    reference_kl: float
    opsd: float
    total: float
    mean_reward: float
    distillation_tokens: int


def joint_loss(grpo: float, opsd: float, lambda_opsd: float = 0.05) -> float:
    if lambda_opsd < 0:
        raise ValueError("lambda_opsd must be non-negative")
    return grpo + lambda_opsd * opsd


def compute_mapd_loss(
    group: list[RolloutLossInput],
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    beta: float = 0.0,
    lambda_opsd: float = 0.05,
) -> LossBreakdown:
    if not group:
        raise ValueError("a training group cannot be empty")
    if min(clip_low, clip_high, beta, lambda_opsd) < 0:
        raise ValueError("loss coefficients must be non-negative")
    advantages = group_relative_advantages([item.reward for item in group])
    rl_loss = grpo_loss(
        group,
        advantages,
        clip_low=clip_low,
        clip_high=clip_high,
        beta=beta,
    )
    trajectory_reference_kls = []
    for rollout in group:
        active = [
            reference_kl
            for reference_kl, enabled in zip(
                rollout.reference_token_kls, rollout.action_mask
            )
            if enabled
        ]
        trajectory_reference_kls.append(sum(active) / len(active) if active else 0.0)
    reference_kl = sum(trajectory_reference_kls) / len(trajectory_reference_kls)
    distillation = []
    for rollout in group:
        if rollout.privileged_log_distributions is None:
            continue
        for student, privileged, active in zip(
            rollout.student_log_distributions,
            rollout.privileged_log_distributions,
            rollout.action_mask,
        ):
            if active:
                distillation.append(reverse_kl(student, privileged))
    opsd_loss = sum(distillation) / len(distillation) if distillation else 0.0
    return LossBreakdown(
        grpo=rl_loss,
        reference_kl=reference_kl,
        opsd=opsd_loss,
        total=rl_loss + lambda_opsd * opsd_loss,
        mean_reward=sum(item.reward for item in group) / len(group),
        distillation_tokens=len(distillation),
    )


class OnPolicyTrainingBackend(Protocol):
    def rollout_group(self, sample: QASample, group_size: int, max_turns: int) -> list[AgentTrajectory]: ...

    def optimize_group(
        self,
        sample: QASample,
        trajectories: list[AgentTrajectory],
        privileged_information: PrivilegedInformation | None,
    ) -> LossBreakdown: ...

    def save_checkpoint(self, step: int) -> None: ...


@dataclass(frozen=True)
class TrainingStepResult:
    example_id: str
    rewards: list[float]
    pi_source: str
    loss: LossBreakdown


class MAPDTrainingLoop:
    def __init__(self, backend: OnPolicyTrainingBackend, *, group_size: int, max_turns: int):
        self.backend = backend
        self.group_size = group_size
        self.max_turns = max_turns

    def train_example(self, artifact: SynthesisArtifact) -> TrainingStepResult:
        trajectories = self.backend.rollout_group(artifact.sample, self.group_size, self.max_turns)
        if len(trajectories) != self.group_size:
            raise ValueError("backend returned a rollout group with the wrong size")
        privileged = select_privileged_information(artifact, trajectories)
        loss = self.backend.optimize_group(artifact.sample, trajectories, privileged)
        return TrainingStepResult(
            artifact.sample.id,
            [item.reward for item in trajectories],
            privileged.source if privileged else "disabled",
            loss,
        )
