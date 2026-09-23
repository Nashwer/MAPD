#!/usr/bin/env python3
"""Minimal headless vLLM smoke test for the local MAPD student model."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    default_model = Path.home() / "models" / "Qwen3-1.7B"
    supplied_model = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    model_path = Path(
        supplied_model or os.environ.get("MAPD_MODEL_PATH", str(default_model))
    ).expanduser()

    config_path = model_path / "config.json"
    if not config_path.is_file():
        print(f"ERROR: model config not found: {config_path}", file=sys.stderr)
        print(
            "Set MAPD_MODEL_PATH or pass the model directory as the second argument.",
            file=sys.stderr,
        )
        return 2

    import torch
    from vllm import LLM, SamplingParams

    print(f"model: {model_path}", flush=True)
    print(f"cuda: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        return 3
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print("loading model...", flush=True)

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.60,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "In which city is the Eiffel Tower located? "
                    "Finish with <answer>short answer</answer>."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=128),
    )[0].outputs[0].text

    print("\n===== MODEL OUTPUT =====")
    print(result.strip())
    print("========================")
    print("MODEL SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
