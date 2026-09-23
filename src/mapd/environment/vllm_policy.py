from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class VLLMStudentPolicy:
    """Adapter from vLLM's offline engine to the search environment policy contract."""

    engine: Any
    tokenizer: Any
    sampling_params_type: Callable[..., Any]
    temperature: float = 0.0

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
        )
        request_output = self.engine.generate([prompt], params)[0]
        return request_output.outputs[0].text.strip()
