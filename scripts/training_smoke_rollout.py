#!/usr/bin/env python3
"""Collect a small on-policy vLLM rollout group with exact token metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import read_jsonl, write_jsonl
from mapd.mas.schema import SynthesisArtifact
from mapd.retrieval.wiki18 import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/smoke/artifacts.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/corpus.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train_smoke/rollouts.jsonl"))
    parser.add_argument("--example-id", default="smoke-2")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.group_size < 1:
        raise ValueError("group-size must be positive")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found under {args.model}")
    artifacts = read_jsonl(args.artifacts, SynthesisArtifact)
    artifact = next((item for item in artifacts if item.sample.id == args.example_id), None)
    if artifact is None:
        raise ValueError(f"example {args.example_id!r} is absent from {args.artifacts}")

    required_search_prompt = (
        AGENT_SYSTEM_PROMPT
        + "\nFor this training acceptance run, the first response MUST contain exactly "
        "one <search>query</search> action. After receiving information, finish with "
        "exactly one <answer>short answer</answer> action."
    )
    policy = VLLMStudentPolicy.from_model(args.model)
    policy.temperature = args.temperature
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl(args.corpus),
        top_k=1,
        max_turns=2,
        system_prompt=required_search_prompt,
        require_search=True,
    )
    trajectories = [
        environment.rollout(
            artifact.sample,
            policy,
            max_new_tokens=args.max_new_tokens,
        )
        for _ in range(args.group_size)
    ]
    for trajectory in trajectories:
        if not trajectory.turns or trajectory.turns[0].action != "search":
            raise RuntimeError("training acceptance rollout did not begin with a real search action")
        if any(turn.action == "invalid" for turn in trajectory.turns):
            raise RuntimeError("training acceptance rollout contains an invalid action")
        if any("<information>" in turn.model_output.lower() for turn in trajectory.turns):
            raise RuntimeError("model attempted to generate the environment information block")
        for turn in trajectory.turns:
            if turn.token_ids is None or turn.token_log_probs is None:
                raise RuntimeError("vLLM did not return exact token ids and selected-token log-probabilities")

    write_jsonl(args.output, trajectories)
    summary = {
        "example_id": artifact.sample.id,
        "group_size": len(trajectories),
        "rewards": [item.reward for item in trajectories],
        "response_tokens": [
            sum(len(turn.token_ids or []) for turn in item.turns) for item in trajectories
        ],
        "searched": [any(turn.action == "search" for turn in item.turns) for item in trajectories],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("TRAINING ROLLOUT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
