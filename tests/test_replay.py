from copy import deepcopy

import pytest
from pydantic import ValidationError

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory, AgentTurn
from mapd.trainer.opsd import PrivilegedInformation
from mapd.trainer.replay import build_token_replay


class RecordingTokenizer:
    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.rendered.append(deepcopy(messages))
        return {
            "input_ids": [[1, len(messages), len(messages[-1]["content"]) % 17 + 2]]
        }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) % 31 for character in text]


def test_token_replay_keeps_response_ids_and_privileged_context_separate():
    sample = QASample(id="q1", question="Where?", answers=["Paris"])
    trajectory = AgentTrajectory(
        example_id="q1",
        prompt="Where?",
        reward=1.0,
        terminated=True,
        final_answer="Paris",
        turns=[
            AgentTurn(
                turn_index=1,
                model_output="<search>place</search>",
                action="search",
                query="place",
                observation="<information>Paris</information>",
                token_ids=[10, 11],
                token_log_probs=[-0.1, -0.2],
            ),
            AgentTurn(
                turn_index=2,
                model_output="<answer>Paris</answer>",
                action="answer",
                answer="Paris",
                token_ids=[12, 13],
                token_log_probs=[-0.3, -0.4],
            ),
        ],
    )
    tokenizer = RecordingTokenizer()

    replay = build_token_replay(
        sample,
        trajectory,
        tokenizer,
        PrivilegedInformation("protocol", "PRIVATE PROTOCOL"),
    )

    assert [turn.response_ids for turn in replay] == [[10, 11], [12, 13]]
    assert replay[0].old_log_probs == [-0.1, -0.2]
    assert "PRIVATE PROTOCOL" not in tokenizer.rendered[0][1]["content"]
    assert tokenizer.rendered[1][1]["content"].endswith("PRIVATE PROTOCOL")
    assert tokenizer.rendered[2][-2]["content"] == "<search>place</search>"
    assert tokenizer.rendered[2][-1]["content"] == "<information>Paris</information>"
    assert replay[1].student_prefix_ids != replay[0].student_prefix_ids


def test_agent_turn_rejects_misaligned_rollout_log_probs():
    with pytest.raises(ValidationError):
        AgentTurn(
            turn_index=1,
            model_output="<answer>x</answer>",
            action="answer",
            token_ids=[1, 2],
            token_log_probs=[-0.1],
        )


def test_tokenizer_mapping_must_contain_input_ids():
    class BrokenTokenizer(RecordingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {"attention_mask": [[1, 1]]}

    sample = QASample(id="q1", question="Where?", answers=["Paris"])
    trajectory = AgentTrajectory(
        example_id="q1",
        prompt="Where?",
        reward=0.0,
        turns=[AgentTurn(turn_index=1, model_output="x", action="invalid", token_ids=[3])],
    )
    with pytest.raises(ValueError, match="no input_ids"):
        build_token_replay(sample, trajectory, BrokenTokenizer(), None)
