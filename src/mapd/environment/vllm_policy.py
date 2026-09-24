from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ACTION_STOP_STRINGS = ["</search>", "</answer>"]


@dataclass
class VLLMStudentPolicy:
    """Adapter from vLLM's offline engine to the search environment policy contract."""

    engine: Any
    tokenizer: Any
    sampling_params_type: Callable[..., Any]
    temperature: float = 0.0
    last_token_ids: list[int] | None = field(default=None, init=False)
    last_token_log_probs: list[float] | None = field(default=None, init=False)

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        *,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.60,
        enforce_eager: bool = True,
    ) -> "VLLMStudentPolicy":
        from vllm import LLM, SamplingParams

        engine = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        return cls(
            engine=engine,
            tokenizer=engine.get_tokenizer(),
            sampling_params_type=SamplingParams,
        )

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = self.sampling_params_type(
            temperature=self.temperature,
            max_tokens=max_new_tokens,
            logprobs=1,
            stop=ACTION_STOP_STRINGS,
            include_stop_str_in_output=True,
        )
        request_output = self.engine.generate([prompt], params)[0]
        generation = request_output.outputs[0]
        token_ids = getattr(generation, "token_ids", None)
        self.last_token_ids = [int(token_id) for token_id in token_ids] if token_ids is not None else None
        self.last_token_log_probs = _selected_token_log_probs(
            self.last_token_ids,
            getattr(generation, "logprobs", None),
        )
        return generation.text


def _selected_token_log_probs(
    token_ids: list[int] | None, candidates_by_position: Any
) -> list[float] | None:
    if token_ids is None or candidates_by_position is None:
        return None
    if len(token_ids) != len(candidates_by_position):
        return None
    selected: list[float] = []
    for token_id, candidates in zip(token_ids, candidates_by_position):
        if candidates is None or token_id not in candidates:
            return None
        value = candidates[token_id]
        selected.append(float(getattr(value, "logprob", value)))
    return selected
