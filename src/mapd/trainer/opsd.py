from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pydantic import BaseModel, model_validator

from mapd.environment.schema import AgentTrajectory
from mapd.environment.search_env import generated_trajectory_text
from mapd.mas.schema import SynthesisArtifact
from mapd.protocol.schema import StructuredProtocol


class PrivilegedSequence(BaseModel):
    student_prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[bool]
    privileged_context_ids: list[int]

    @model_validator(mode="after")
    def validate_alignment(self) -> "PrivilegedSequence":
        if len(self.response_ids) != len(self.response_mask):
            raise ValueError("response_ids and response_mask must have identical length")
        return self

    @property
    def teacher_sequence_ids(self) -> list[int]:
        # Paper notation: privileged branch state is (x, p, y_<t).
        return self.student_prompt_ids + self.privileged_context_ids + self.response_ids


def protocol_context(protocol: StructuredProtocol) -> str:
    return "Privileged MAPD protocol (training only):\n" + json.dumps(
        protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
    )


@dataclass(frozen=True)
class PrivilegedInformation:
    source: str
    content: str


def select_privileged_information(
    artifact: SynthesisArtifact, student_group: list[AgentTrajectory]
) -> PrivilegedInformation | None:
    if artifact.quality.passed and artifact.protocol is not None:
        return PrivilegedInformation("protocol", protocol_context(artifact.protocol))
    successful = next((trajectory for trajectory in student_group if trajectory.reward == 1.0), None)
    if successful is None:
        return None
    return PrivilegedInformation(
        "self_rollout",
        "Privileged successful student rollout (training only):\n" + generated_trajectory_text(successful),
    )


def reverse_kl(student_log_probs: list[float], privileged_log_probs: list[float]) -> float:
    if len(student_log_probs) != len(privileged_log_probs) or not student_log_probs:
        raise ValueError("student and privileged distributions must share a non-empty vocabulary")
    return sum(
        math.exp(student) * (student - privileged)
        for student, privileged in zip(student_log_probs, privileged_log_probs)
    )


def torch_reverse_kl(student_logits, privileged_logits, action_mask):
    import torch

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    privileged_log_probs = torch.log_softmax(privileged_logits.detach(), dim=-1)
    token_kl = (student_log_probs.exp() * (student_log_probs - privileged_log_probs)).sum(-1)
    mask = action_mask.to(token_kl.dtype)
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
