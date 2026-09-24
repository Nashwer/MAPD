from __future__ import annotations

import pytest

from mapd.environment.schema import AgentTrajectory, AgentTurn
from mapd.mas.schema import SynthesisArtifact
from scripts.protocol_train_smoke_optimize import _group_rollouts, _validate_inputs


def _trajectory(example_id: str) -> AgentTrajectory:
    return AgentTrajectory(
        example_id=example_id,
        prompt="question",
        turns=[
            AgentTurn(
                turn_index=1,
                model_output="<answer>x</answer>",
                action="answer",
                answer="x",
                token_ids=[1],
                token_log_probs=[-0.1],
            )
        ],
        final_answer="x",
        reward=0.0,
        terminated=True,
    )


def test_real_protocol_training_requires_aligned_groups(valid_artifact: SynthesisArtifact):
    groups = _group_rollouts([_trajectory(valid_artifact.sample.id) for _ in range(2)])
    _validate_inputs([valid_artifact], groups)


def test_real_protocol_training_rejects_missing_groups(valid_artifact: SynthesisArtifact):
    with pytest.raises(ValueError, match="rollout/artifact mismatch"):
        _validate_inputs([valid_artifact], {})


@pytest.fixture
def valid_artifact() -> SynthesisArtifact:
    return SynthesisArtifact.model_validate(
        {
            "sample": {
                "id": "sample-1",
                "question": "Who directed The Winter Guest?",
                "answers": ["Alan Rickman"],
            },
            "exploration": {
                "example_id": "sample-1",
                "question": "Who directed The Winter Guest?",
                "task_type": "single_hop",
                "subtasks": [],
                "searches": [],
                "candidate_answer": "Alan Rickman",
                "success": True,
            },
            "protocol": {
                "task_type": "single_hop",
                "reasoning_plan": ["Find the director."],
                "grounding_facts": ["The Winter Guest was directed by Alan Rickman."],
                "answer": "Alan Rickman",
                "answer_grounded": True,
            },
            "raw_protocol": {},
            "quality": {
                "example_id": "sample-1",
                "passed": True,
                "checks": {
                    "schema": True,
                    "em_consistency": True,
                    "extractive_grounding": True,
                    "no_answer_leak": True,
                },
                "errors": [],
                "pi_source": "protocol",
            },
        }
    )
