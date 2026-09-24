from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory, AgentTurn
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT
from mapd.trainer.opsd import PrivilegedInformation


@dataclass(frozen=True)
class TokenReplayTurn:
    """One generated action with aligned student and privileged prefixes."""

    student_prefix_ids: list[int]
    privileged_prefix_ids: list[int] | None
    response_ids: list[int]
    old_log_probs: list[float] | None


def build_token_replay(
    sample: QASample,
    trajectory: AgentTrajectory,
    tokenizer: Any,
    privileged_information: PrivilegedInformation | None,
    *,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> list[TokenReplayTurn]:
    if trajectory.example_id != sample.id:
        raise ValueError("trajectory and sample example ids do not match")

    effective_system_prompt = trajectory.system_prompt or system_prompt
    student_messages = _initial_messages(sample.question, effective_system_prompt, None)
    privileged_messages = (
        _initial_messages(
            sample.question,
            effective_system_prompt,
            privileged_information.content,
        )
        if privileged_information is not None
        else None
    )
    replay: list[TokenReplayTurn] = []
    for turn in trajectory.turns:
        response_ids = _response_ids(turn, tokenizer)
        if not response_ids:
            raise ValueError(f"turn {turn.turn_index} has no response tokens")
        replay.append(
            TokenReplayTurn(
                student_prefix_ids=_render_prefix(tokenizer, student_messages),
                privileged_prefix_ids=(
                    _render_prefix(tokenizer, privileged_messages)
                    if privileged_messages is not None
                    else None
                ),
                response_ids=response_ids,
                old_log_probs=(
                    list(turn.token_log_probs)
                    if turn.token_log_probs is not None
                    else None
                ),
            )
        )
        _append_turn(student_messages, turn)
        if privileged_messages is not None:
            _append_turn(privileged_messages, turn)
    return replay


def _initial_messages(
    question: str, system_prompt: str, privileged_content: str | None
) -> list[dict[str, str]]:
    user_content = question
    if privileged_content:
        # Paper state: (x, p, y_<t>). The PI is appended after x and is never
        # added to the rollout branch or the serialized public trajectory.
        user_content += "\n\n" + privileged_content
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _render_prefix(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token_id) for token_id in rendered]


def _response_ids(turn: AgentTurn, tokenizer: Any) -> list[int]:
    if turn.token_ids is not None:
        return list(turn.token_ids)
    encoded = tokenizer.encode(turn.model_output, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def _append_turn(messages: list[dict[str, str]], turn: AgentTurn) -> None:
    messages.append({"role": "assistant", "content": turn.model_output})
    if turn.action == "search":
        messages.append(
            {
                "role": "user",
                "content": turn.observation or "<information>\n\n</information>",
            }
        )
    elif turn.action == "invalid":
        messages.append(
            {"role": "user", "content": "Use one <search> or <answer> action."}
        )
