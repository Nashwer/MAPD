#!/usr/bin/env python3
"""Run one real multi-turn Qwen -> search -> observation -> answer trajectory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mapd.data.schema import QASample
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import write_jsonl
from mapd.retrieval.wiki18 import BM25Retriever


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    default_model = Path.home() / "models" / "Qwen3-1.7B"
    supplied_model = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    model_path = Path(
        supplied_model or os.environ.get("MAPD_MODEL_PATH", str(default_model))
    ).expanduser()

    if not (model_path / "config.json").is_file():
        print(f"ERROR: model config not found: {model_path / 'config.json'}", file=sys.stderr)
        return 2

    corpus_path = project_root / "tests" / "fixtures" / "corpus.jsonl"
    output_path = project_root / "artifacts" / "agent_smoke" / "trajectory.jsonl"
    required_search_prompt = (
        AGENT_SYSTEM_PROMPT
        + "\nFor this verification run, your first response MUST contain exactly one "
        "<search>query</search> action and no answer. After receiving information, "
        "finish with exactly one <answer>short answer</answer> action."
    )

    print(f"model: {model_path}", flush=True)
    print("loading vLLM policy...", flush=True)
    policy = VLLMStudentPolicy.from_model(model_path)
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl(corpus_path),
        top_k=1,
        max_turns=3,
        system_prompt=required_search_prompt,
    )
    sample = QASample(
        id="agent-smoke-1",
        question="Who directed the 1997 film The Winter Guest?",
        answers=["Alan Rickman"],
        split="smoke",
        data_source="fixture",
    )

    trajectory = environment.rollout(sample, policy, max_new_tokens=128)
    write_jsonl(output_path, [trajectory])

    print("\n===== AGENT TRAJECTORY =====")
    for turn in trajectory.turns:
        print(f"turn={turn.turn_index} action={turn.action}")
        print(f"model: {turn.model_output}")
        if turn.query:
            print(f"query: {turn.query}")
        if turn.observation:
            print(f"observation: {turn.observation}")
    print(f"final_answer: {trajectory.final_answer}")
    print(f"exact_match_reward: {trajectory.reward}")
    print(f"artifact: {output_path}")
    print("============================")

    searched = any(turn.action == "search" for turn in trajectory.turns)
    if not searched:
        print("AGENT SMOKE FAILED: the model did not call search", file=sys.stderr)
        return 3
    if not trajectory.terminated or trajectory.reward != 1.0:
        print("AGENT SMOKE FAILED: no correct terminal answer", file=sys.stderr)
        return 4
    print("AGENT SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
