#!/usr/bin/env python3
"""Run the paper's multi-hop appendix examples through the real search agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapd.data.schema import QASample
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import write_jsonl
from mapd.retrieval.retriever_server import HTTPRetriever


PAPER_EXAMPLES = [
    {
        "appendix_example": 2,
        "expected_hops": 3,
        "sample": QASample(
            id="mapd-paper-example-2",
            question=(
                "Which English actor and director who was involved with the 1997 film "
                "The Winter Guest also appeared in Die Hard?"
            ),
            answers=["Alan Rickman"],
            split="paper_appendix",
            data_source="mapd_paper",
        ),
    },
    {
        "appendix_example": 3,
        "expected_hops": 3,
        "sample": QASample(
            id="mapd-paper-example-3",
            question=(
                "What year was the barber surgeon who served the last French monarch "
                "of the House of Valois born?"
            ),
            answers=["1510"],
            split="paper_appendix",
            data_source="mapd_paper",
        ),
    },
    {
        "appendix_example": 4,
        "expected_hops": 3,
        "sample": QASample(
            id="mapd-paper-example-4",
            question=(
                "The band famous for the single Waterfalls had a singer with the "
                "nickname Left Eye, who was born on what day?"
            ),
            answers=["May 27, 1971"],
            split="paper_appendix",
            data_source="mapd_paper",
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAPD Appendix A examples with the real Qwen/search loop"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/paper_examples_action_v3")
    )
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollouts_per_example < 1:
        raise ValueError("rollouts-per-example must be positive")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found under {args.model}")

    policy = VLLMStudentPolicy.from_model(
        args.model,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
    )
    policy.temperature = args.temperature
    environment = AgenticSearchEnvironment(
        HTTPRetriever(args.retriever_url, timeout_seconds=120),
        top_k=args.top_k,
        max_turns=args.max_turns,
        system_prompt=AGENT_SYSTEM_PROMPT,
        require_search=False,
    )
    trajectories = []
    diagnostics = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in PAPER_EXAMPLES:
        sample = entry["sample"]
        example_trajectories = [
            environment.rollout(sample, policy, max_new_tokens=args.max_new_tokens)
            for _ in range(args.rollouts_per_example)
        ]
        trajectories.extend(example_trajectories)
        search_depths = [
            sum(turn.action == "search" for turn in trajectory.turns)
            for trajectory in example_trajectories
        ]
        diagnostic = {
            "appendix_example": entry["appendix_example"],
            "example_id": sample.id,
            "question": sample.question,
            "gold_answers": sample.answers,
            "expected_reasoning_hops": entry["expected_hops"],
            "search_depths": search_depths,
            "answers": [item.final_answer for item in example_trajectories],
            "rewards": [item.reward for item in example_trajectories],
            "multi_search_rollouts": sum(depth >= 2 for depth in search_depths),
        }
        diagnostics.append(diagnostic)
        write_jsonl(args.output_dir / "rollouts.jsonl", trajectories)
        print(json.dumps(diagnostic, ensure_ascii=False), flush=True)

    total = len(trajectories)
    manifest = {
        "schema": "mapd-paper-examples-smoke-v1",
        "source": "MAPD paper Appendix A, Examples 2-4",
        "model": str(args.model.resolve()),
        "retriever_url": args.retriever_url,
        "rollouts_per_example": args.rollouts_per_example,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_turns": args.max_turns,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "trajectories": total,
        "terminated": sum(item.terminated for item in trajectories),
        "exact_match": sum(item.reward for item in trajectories),
        "multi_search_trajectories": sum(
            sum(turn.action == "search" for turn in item.turns) >= 2
            for item in trajectories
        ),
        "examples": diagnostics,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["multi_search_trajectories"] == 0:
        print("PAPER EXAMPLE SMOKE: no multi-search trajectory observed")
    else:
        print("PAPER EXAMPLE MULTI-HOP SEARCH OBSERVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
