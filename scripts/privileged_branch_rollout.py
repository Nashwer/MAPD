#!/usr/bin/env python3
"""Generate diagnostic agent rollouts from MAPD privileged protocol states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any

from mapd.environment.schema import AgentTrajectory
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import read_jsonl, write_jsonl
from mapd.mas.schema import SynthesisArtifact
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.trainer.opsd import select_privileged_information


RUN_SCHEMA = "mapd-privileged-generation-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    artifacts = read_jsonl(args.artifacts, SynthesisArtifact)
    if not artifacts:
        raise ValueError("protocol artifact shard is empty")
    ids = [artifact.sample.id for artifact in artifacts]
    if len(ids) != len(set(ids)):
        raise ValueError("protocol artifact shard contains duplicate example ids")
    privileged = {}
    for artifact in artifacts:
        information = select_privileged_information(artifact, [])
        if information is None:
            raise ValueError(
                f"example {artifact.sample.id!r} has no valid protocol privileged information"
            )
        privileged[artifact.sample.id] = information

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = args.output_dir / "rollouts.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    context_path = args.output_dir / "privileged_contexts.jsonl"
    run_config = _run_config(args)
    completed = _load_completed(
        rollout_path,
        manifest_path,
        run_config,
        set(ids),
        args.rollouts_per_example,
    )
    pending = [artifact for artifact in artifacts if artifact.sample.id not in completed]
    policy = None
    environment = None
    if pending:
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

    for artifact in pending:
        assert policy is not None and environment is not None
        information = privileged[artifact.sample.id]
        group = [
            environment.rollout(
                artifact.sample,
                policy,
                max_new_tokens=args.max_new_tokens,
                privileged_context=information.content,
            )
            for _ in range(args.rollouts_per_example)
        ]
        _validate_group(artifact.sample.id, group, args.rollouts_per_example)
        completed[artifact.sample.id] = group
        _write_outputs(
            artifacts,
            completed,
            privileged,
            rollout_path,
            context_path,
            manifest_path,
            run_config,
        )
        print(
            json.dumps(
                {
                    "event": "privileged_generation_progress",
                    "processed": len(completed),
                    "total": len(artifacts),
                    "example_id": artifact.sample.id,
                    "privileged_source": information.source,
                    "rewards": [trajectory.reward for trajectory in group],
                    "search_depths": [
                        sum(turn.action == "search" for turn in trajectory.turns)
                        for trajectory in group
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _write_outputs(
        artifacts,
        completed,
        privileged,
        rollout_path,
        context_path,
        manifest_path,
        run_config,
    )
    summary = _summary(artifacts, completed, run_config)
    archive = _write_archive(args.output_dir)
    summary["archive"] = str(archive.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PRIVILEGED GENERATION PROBE OK")
    return 0


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "mode": "diagnostic_generation_from_privileged_state",
        "paper_faithful_training_branch": False,
        "note": (
            "The paper OPSD branch scores ordinary-rollout response tokens under (x,p,y_<t>). "
            "This probe samples new actions from that privileged state only for inspection."
        ),
        "model": str(args.model.resolve()),
        "artifacts": str(args.artifacts.resolve()),
        "artifacts_sha256": _sha256(args.artifacts),
        "retriever_url": args.retriever_url,
        "rollouts_per_example": args.rollouts_per_example,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_turns": args.max_turns,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
    }


def _load_completed(
    rollout_path: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
    allowed_ids: set[str],
    group_size: int,
) -> dict[str, list[AgentTrajectory]]:
    if not rollout_path.exists() and not manifest_path.exists():
        return {}
    if not rollout_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("privileged rollout checkpoint is incomplete; use a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_config") != run_config:
        raise RuntimeError("privileged rollout configuration changed; use a new output directory")
    completed: dict[str, list[AgentTrajectory]] = {}
    for trajectory in read_jsonl(rollout_path, AgentTrajectory):
        if trajectory.example_id not in allowed_ids:
            raise ValueError(f"unexpected rollout example {trajectory.example_id!r}")
        completed.setdefault(trajectory.example_id, []).append(trajectory)
    for example_id, group in completed.items():
        _validate_group(example_id, group, group_size)
    return completed


def _validate_group(
    example_id: str, group: list[AgentTrajectory], expected_size: int
) -> None:
    if len(group) != expected_size:
        raise ValueError(f"example {example_id!r} has an incomplete rollout group")
    for trajectory in group:
        if trajectory.example_id != example_id:
            raise ValueError("rollout group contains multiple example ids")
        if not trajectory.turns:
            raise ValueError("privileged rollout contains no turns")
        for turn in trajectory.turns:
            if turn.token_ids is None or turn.token_log_probs is None:
                raise RuntimeError("vLLM did not return exact rollout token metadata")


def _ordered_trajectories(
    artifacts: list[SynthesisArtifact],
    completed: dict[str, list[AgentTrajectory]],
) -> list[AgentTrajectory]:
    return [
        trajectory
        for artifact in artifacts
        for trajectory in completed.get(artifact.sample.id, [])
    ]


def _summary(
    artifacts: list[SynthesisArtifact],
    completed: dict[str, list[AgentTrajectory]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    trajectories = _ordered_trajectories(artifacts, completed)
    exact = sum(trajectory.reward for trajectory in trajectories)
    return {
        "run_config": run_config,
        "examples": len(artifacts),
        "completed_examples": len(completed),
        "trajectories": len(trajectories),
        "privileged_protocols": len(artifacts),
        "terminated": sum(trajectory.terminated for trajectory in trajectories),
        "exact_match": exact,
        "accuracy": exact / len(trajectories) if trajectories else None,
        "search_turns": sum(
            turn.action == "search"
            for trajectory in trajectories
            for turn in trajectory.turns
        ),
        "multi_search_trajectories": sum(
            sum(turn.action == "search" for turn in trajectory.turns) >= 2
            for trajectory in trajectories
        ),
        "direct_answer_trajectories": sum(
            all(turn.action != "search" for turn in trajectory.turns)
            for trajectory in trajectories
        ),
        "examples_detail": [
            {
                "example_id": artifact.sample.id,
                "gold_answers": artifact.sample.answers,
                "protocol_variant": artifact.metadata.get("protocol_variant"),
                "answers": [
                    trajectory.final_answer
                    for trajectory in completed.get(artifact.sample.id, [])
                ],
                "rewards": [
                    trajectory.reward for trajectory in completed.get(artifact.sample.id, [])
                ],
                "search_depths": [
                    sum(turn.action == "search" for turn in trajectory.turns)
                    for trajectory in completed.get(artifact.sample.id, [])
                ],
            }
            for artifact in artifacts
        ],
    }


def _write_outputs(
    artifacts: list[SynthesisArtifact],
    completed: dict[str, list[AgentTrajectory]],
    privileged: dict[str, Any],
    rollout_path: Path,
    context_path: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
) -> None:
    write_jsonl(rollout_path, _ordered_trajectories(artifacts, completed))
    write_jsonl(
        context_path,
        [
            {
                "example_id": artifact.sample.id,
                "privileged_source": privileged[artifact.sample.id].source,
                "privileged_content": privileged[artifact.sample.id].content,
            }
            for artifact in artifacts
        ],
    )
    payload = _summary(artifacts, completed, run_config)
    temporary = manifest_path.with_name(manifest_path.name + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive(output_dir: Path) -> Path:
    archive = output_dir.with_suffix(".tar.gz")
    temporary = archive.with_name(archive.name + ".building")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz") as bundle:
        for name in ("manifest.json", "rollouts.jsonl", "privileged_contexts.jsonl"):
            path = output_dir / name
            if path.is_file():
                bundle.add(path, arcname=name)
    os.replace(temporary, archive)
    return archive


if __name__ == "__main__":
    raise SystemExit(main())
