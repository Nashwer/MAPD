from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mapd.data.schema import QASample
from mapd.environment.parser import parse_action
from mapd.environment.schema import AgentTrajectory, AgentTurn
from mapd.retrieval.base import Retriever
from mapd.retrieval.schema import RetrievedPassage
from mapd.reward.exact_match import exact_match


AGENT_SYSTEM_PROMPT = """Answer the question by reasoning and using the search tool when needed.
Issue a query as <search>query</search>. The environment will return <information>...</information>.
When ready, finish with <answer>short answer</answer>.
The answer tag must contain only the minimal answer span, never an explanatory sentence.
For example, use <answer>Paris</answer>, not <answer>The city is Paris.</answer>."""


class StudentPolicy(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...


@dataclass
class ScriptedPolicy:
    outputs: list[str]
    cursor: int = field(default=0, init=False)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        del messages, max_new_tokens
        if self.cursor >= len(self.outputs):
            return "<answer></answer>"
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output


class AgenticSearchEnvironment:
    def __init__(
        self,
        retriever: Retriever,
        *,
        top_k: int = 3,
        max_turns: int = 4,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    def rollout(
        self, sample: QASample, policy: StudentPolicy, *, max_new_tokens: int = 512
    ) -> AgentTrajectory:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": sample.question},
        ]
        turns: list[AgentTurn] = []
        final_answer: str | None = None
        terminated = False
        for turn_index in range(1, self.max_turns + 1):
            output = policy.generate(messages, max_new_tokens)
            token_ids = getattr(policy, "last_token_ids", None)
            token_log_probs = getattr(policy, "last_token_log_probs", None)
            trace = {
                "token_ids": list(token_ids) if token_ids is not None else None,
                "token_log_probs": (
                    list(token_log_probs) if token_log_probs is not None else None
                ),
            }
            action = parse_action(output)
            if action.kind == "answer":
                final_answer = action.value
                terminated = True
                turns.append(
                    AgentTurn(
                        turn_index=turn_index,
                        model_output=output,
                        action="answer",
                        answer=final_answer,
                        **trace,
                    )
                )
                break
            if action.kind == "search":
                query = action.value or ""
                observation = format_observation(self.retriever.search(query, self.top_k) if query else [])
                turns.append(
                    AgentTurn(
                        turn_index=turn_index,
                        model_output=output,
                        action="search",
                        query=query,
                        observation=observation,
                        **trace,
                    )
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": output},
                        {"role": "user", "content": observation},
                    ]
                )
                continue
            turns.append(
                AgentTurn(
                    turn_index=turn_index,
                    model_output=output,
                    action="invalid",
                    **trace,
                )
            )
            messages.extend(
                [
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": "Use one <search> or <answer> action."},
                ]
            )
        return AgentTrajectory(
            example_id=sample.id,
            prompt=sample.question,
            system_prompt=self.system_prompt,
            turns=turns,
            final_answer=final_answer,
            reward=float(exact_match(final_answer, sample.answers)),
            terminated=terminated,
        )


def format_observation(passages: list[RetrievedPassage]) -> str:
    body = "\n\n".join(f"[{item.id}] {item.contents}" for item in passages)
    return f"<information>\n{body}\n</information>"


def generated_trajectory_text(trajectory: AgentTrajectory) -> str:
    return "\n".join(turn.model_output for turn in trajectory.turns)
