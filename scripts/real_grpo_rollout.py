#!/usr/bin/env python3
"""Find a real on-policy rollout group with both correct and incorrect rewards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from mapd.data.schema import QASample
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import read_jsonl, write_jsonl
from mapd.retrieval.retriever_server import HTTPRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/real_grpo_smoke"))
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("group-size must be at least 2")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found under {args.model}")
    samples = read_jsonl(args.qa, QASample)
    if not samples:
        raise ValueError("QA input is empty")
    random.Random(args.seed).shuffle(samples)
    candidates = samples[: args.candidate_limit]

    prompt = (
        AGENT_SYSTEM_PROMPT
        + "\nFor this acceptance run, begin with one <search>query</search> action. "
        "Use the returned passages, then finish with one <answer>short answer</answer>."
    )
    policy = VLLMStudentPolicy.from_model(args.model)
    policy.temperature = args.temperature
    environment = AgenticSearchEnvironment(
        HTTPRetriever(args.retriever_url, timeout_seconds=120),
        top_k=args.top_k,
        max_turns=args.max_turns,
        system_prompt=prompt,
    )

    diagnostics = []
    for candidate_index, sample in enumerate(candidates, start=1):
        print(
            f"candidate {candidate_index}/{len(candidates)}: {sample.id} {sample.question}",
            flush=True,
        )
        trajectories = [
            environment.rollout(sample, policy, max_new_tokens=args.max_new_tokens)
            for _ in range(args.group_size)
        ]
        rewards = [item.reward for item in trajectories]
        searches = [any(turn.action == "search" for turn in item.turns) for item in trajectories]
        diagnostics.append(
            {"id": sample.id, "rewards": rewards, "searched": searches}
        )
        if len(set(rewards)) < 2:
            continue
        for trajectory in trajectories:
            for turn in trajectory.turns:
                if turn.token_ids is None or turn.token_log_probs is None:
                    raise RuntimeError("vLLM did not return exact rollout token metadata")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "sample.jsonl", [sample])
        write_jsonl(args.output_dir / "rollouts.jsonl", trajectories)
        summary = {
            "sample": sample.model_dump(mode="json"),
            "group_size": args.group_size,
            "rewards": rewards,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_variance_nonzero": True,
            "searched": searches,
            "top_k": args.top_k,
            "retriever_url": args.retriever_url,
            "attempts": diagnostics,
        }
        (args.output_dir / "rollout-manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("NONZERO REWARD VARIANCE ROLLOUT OK")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failed-rollout-diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError(
        "no mixed-reward group found; increase --candidate-limit or adjust --temperature"
    )


if __name__ == "__main__":
    raise SystemExit(main())
