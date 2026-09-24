#!/usr/bin/env python3
"""Collect real student rollouts for a generated protocol shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mapd.environment.schema import AgentTrajectory
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import read_jsonl, write_jsonl
from mapd.mas.schema import SynthesisArtifact
from mapd.retrieval.retriever_server import HTTPRetriever


RUN_SCHEMA = "mapd-protocol-train-rollout-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out the student on every sample in a real protocol shard"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("group-size must be at least 2")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found under {args.model}")
    artifacts = read_jsonl(args.artifacts, SynthesisArtifact)
    if not artifacts:
        raise ValueError("protocol artifact shard is empty")
    _require_unique_ids(artifacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = args.output_dir / "rollouts.jsonl"
    manifest_path = args.output_dir / "rollout-manifest.json"
    run_config = _run_config(args)
    completed = _load_completed_rollouts(
        rollout_path,
        manifest_path,
        run_config,
        artifacts,
        args.group_size,
    )
    trajectories = [item for group in completed.values() for item in group]
    pending = [item for item in artifacts if item.sample.id not in completed]

    policy = None
    environment = None
    if pending:
        prompt = (
            AGENT_SYSTEM_PROMPT
            + "\nFor this MAPD training run, begin with one <search>query</search> action. "
            "Use the returned passages and finish with one <answer>short answer</answer>."
        )
        policy = VLLMStudentPolicy.from_model(args.model)
        policy.temperature = args.temperature
        environment = AgenticSearchEnvironment(
            HTTPRetriever(args.retriever_url, timeout_seconds=120),
            top_k=args.top_k,
            max_turns=args.max_turns,
            system_prompt=prompt,
            require_search=True,
        )

    for artifact in pending:
        assert policy is not None and environment is not None
        group = [
            environment.rollout(
                artifact.sample,
                policy,
                max_new_tokens=args.max_new_tokens,
            )
            for _ in range(args.group_size)
        ]
        _validate_group(artifact.sample.id, group, args.group_size)
        completed[artifact.sample.id] = group
        trajectories = [
            trajectory
            for candidate in artifacts
            for trajectory in completed.get(candidate.sample.id, [])
        ]
        write_jsonl(rollout_path, trajectories)
        _write_manifest(manifest_path, run_config, artifacts, completed)
        print(
            json.dumps(
                {
                    "event": "protocol_train_rollout_progress",
                    "processed": len(completed),
                    "total": len(artifacts),
                    "example_id": artifact.sample.id,
                    "protocol_passed": artifact.quality.passed,
                    "rewards": [item.reward for item in group],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _write_manifest(manifest_path, run_config, artifacts, completed)
    print(
        json.dumps(
            {
                "examples": len(artifacts),
                "passed_protocols": sum(item.quality.passed for item in artifacts),
                "rollouts": len(trajectories),
                "mixed_reward_groups": sum(
                    len({item.reward for item in group}) > 1 for group in completed.values()
                ),
                "verified_tool_trajectories": sum(
                    _is_verified_tool_trajectory(item)
                    for group in completed.values()
                    for item in group
                ),
                "output": str(rollout_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("REAL PROTOCOL ROLLOUT OK")
    return 0


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "model": str(args.model.resolve()),
        "artifacts": str(args.artifacts.resolve()),
        "artifacts_sha256": _sha256(args.artifacts),
        "retriever_url": args.retriever_url,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "require_search": True,
    }


def _load_completed_rollouts(
    rollout_path: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
    artifacts: list[SynthesisArtifact],
    group_size: int,
) -> dict[str, list[AgentTrajectory]]:
    if not rollout_path.exists() and not manifest_path.exists():
        return {}
    if not rollout_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("rollout checkpoint is incomplete; use a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_config") != run_config:
        raise RuntimeError("rollout checkpoint configuration changed; use a new output directory")
    allowed_ids = {item.sample.id for item in artifacts}
    grouped: dict[str, list[AgentTrajectory]] = {}
    for trajectory in read_jsonl(rollout_path, AgentTrajectory):
        if trajectory.example_id not in allowed_ids:
            raise ValueError(f"unexpected rollout example {trajectory.example_id!r}")
        grouped.setdefault(trajectory.example_id, []).append(trajectory)
    for example_id, group in grouped.items():
        _validate_group(example_id, group, group_size)
    return grouped


def _validate_group(
    example_id: str, trajectories: list[AgentTrajectory], group_size: int
) -> None:
    if len(trajectories) != group_size:
        raise ValueError(f"example {example_id!r} has an incomplete rollout group")
    for trajectory in trajectories:
        if trajectory.example_id != example_id:
            raise ValueError("rollout group contains multiple example ids")
        if not _is_verified_tool_trajectory(trajectory):
            raise RuntimeError(
                f"example {example_id!r} did not produce a strict search-observation-answer trajectory"
            )
        for turn in trajectory.turns:
            if turn.token_ids is None or turn.token_log_probs is None:
                raise RuntimeError("vLLM did not return exact rollout token metadata")


def _is_verified_tool_trajectory(trajectory: AgentTrajectory) -> bool:
    if not trajectory.terminated or trajectory.final_answer is None or not trajectory.turns:
        return False
    if trajectory.turns[0].action != "search":
        return False
    if trajectory.turns[-1].action != "answer":
        return False
    if any(turn.action == "invalid" for turn in trajectory.turns):
        return False
    if any("<information>" in turn.model_output.lower() for turn in trajectory.turns):
        return False
    search_turns = [turn for turn in trajectory.turns if turn.action == "search"]
    return bool(search_turns) and all(
        turn.query and turn.observation and turn.observation.startswith("<information>")
        for turn in search_turns
    )


def _write_manifest(
    path: Path,
    run_config: dict[str, Any],
    artifacts: list[SynthesisArtifact],
    completed: dict[str, list[AgentTrajectory]],
) -> None:
    payload = {
        "run_config": run_config,
        "examples": len(artifacts),
        "passed_protocols": sum(item.quality.passed for item in artifacts),
        "completed_examples": len(completed),
        "completed_ids": [item.sample.id for item in artifacts if item.sample.id in completed],
        "trajectory_count": sum(len(group) for group in completed.values()),
        "verified_tool_trajectories": sum(
            _is_verified_tool_trajectory(trajectory)
            for group in completed.values()
            for trajectory in group
        ),
        "search_turns": sum(
            turn.action == "search"
            for group in completed.values()
            for trajectory in group
            for turn in trajectory.turns
        ),
        "rewards": {
            item.sample.id: [trajectory.reward for trajectory in completed[item.sample.id]]
            for item in artifacts
            if item.sample.id in completed
        },
    }
    temporary = path.with_name(path.name + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _require_unique_ids(artifacts: list[SynthesisArtifact]) -> None:
    ids = [item.sample.id for item in artifacts]
    if len(ids) != len(set(ids)):
        raise ValueError("protocol artifact shard contains duplicate example ids")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
