#!/usr/bin/env python3
"""Run a reproducible public-context student evaluation on held-out QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mapd.data.prepare import canonical_question
from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT, AgenticSearchEnvironment
from mapd.environment.vllm_policy import VLLMStudentPolicy
from mapd.io import read_jsonl, write_jsonl
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.reward.exact_match import exact_match


RUN_SCHEMA = "mapd-heldout-student-eval-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_quotas(counts: Counter[str], target_size: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {key: target_size * value / total for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target_size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_cohort(samples: list[QASample], *, count: int, seed: int) -> list[QASample]:
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(samples):
        raise ValueError(f"requested {count} examples from a {len(samples)}-example held-out set")
    grouped: dict[str, list[QASample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.data_source].append(sample)
    quotas = _allocate_quotas(
        Counter({source: len(items) for source, items in grouped.items()}), count
    )
    rng = random.Random(seed)
    selected: list[QASample] = []
    for source in sorted(grouped):
        candidates = list(grouped[source])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[source]])
    rng.shuffle(selected)
    return selected


def assert_heldout(samples: list[QASample], training_data: Path | None) -> dict[str, int]:
    if not samples:
        raise ValueError("evaluation cohort is empty")
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation cohort contains duplicate IDs")
    if training_data is None:
        return {"id_overlap": 0, "question_overlap": 0}
    training = read_jsonl(training_data, QASample)
    train_ids = {sample.id for sample in training}
    train_questions = {canonical_question(sample.question) for sample in training}
    id_overlap = sum(sample.id in train_ids for sample in samples)
    question_overlap = sum(
        canonical_question(sample.question) in train_questions for sample in samples
    )
    if id_overlap or question_overlap:
        raise RuntimeError(
            "evaluation/train leakage detected: "
            f"id_overlap={id_overlap}, question_overlap={question_overlap}"
        )
    return {"id_overlap": id_overlap, "question_overlap": question_overlap}


def _model_identity(model: Path) -> dict[str, Any]:
    export_manifest = model / "mapd-export-manifest.json"
    if export_manifest.is_file():
        return {
            "path": str(model.resolve()),
            "export_manifest_sha256": _sha256(export_manifest),
        }
    config = model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"model config not found: {config}")
    weights = sorted(
        (path.name, path.stat().st_size)
        for pattern in ("*.safetensors", "*.bin")
        for path in model.glob(pattern)
    )
    return {
        "path": str(model.resolve()),
        "config_sha256": _sha256(config),
        "weight_files": weights,
    }


def summarize(
    cohort: list[QASample], trajectories: list[AgentTrajectory]
) -> dict[str, Any]:
    by_id = {sample.id: sample for sample in cohort}
    if len(trajectories) != len(cohort) or set(by_id) != {
        item.example_id for item in trajectories
    }:
        raise ValueError("trajectory set does not exactly match the evaluation cohort")

    def metrics(items: list[AgentTrajectory]) -> dict[str, Any]:
        count = len(items)
        depths = [sum(turn.action == "search" for turn in item.turns) for item in items]
        answer_matches = [
            exact_match(item.final_answer, by_id[item.example_id].answers) for item in items
        ]
        valid_successes = sum(item.reward == 1.0 for item in items)
        invalid = [any(turn.action == "invalid" for turn in item.turns) for item in items]
        return {
            "examples": count,
            "strict_successes": valid_successes,
            "strict_accuracy": valid_successes / count if count else 0.0,
            "answer_exact_matches": sum(answer_matches),
            "answer_exact_match_rate": sum(answer_matches) / count if count else 0.0,
            "terminated": sum(item.terminated for item in items),
            "completion_rate": sum(item.terminated for item in items) / count if count else 0.0,
            "invalid_trajectories": sum(invalid),
            "invalid_trajectory_rate": sum(invalid) / count if count else 0.0,
            "search_turns": sum(depths),
            "mean_search_depth": sum(depths) / count if count else 0.0,
            "searched_trajectories": sum(depth >= 1 for depth in depths),
            "multi_search_trajectories": sum(depth >= 2 for depth in depths),
            "multi_search_rate": sum(depth >= 2 for depth in depths) / count if count else 0.0,
            "direct_answer_trajectories": sum(depth == 0 for depth in depths),
            "search_depth_histogram": dict(sorted(Counter(depths).items())),
        }

    overall = metrics(trajectories)
    sources: dict[str, list[AgentTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        sources[by_id[trajectory.example_id].data_source].append(trajectory)
    overall["by_source"] = {
        source: metrics(items) for source, items in sorted(sources.items())
    }
    return overall


def _write_manifest(
    path: Path,
    run_config: dict[str, Any],
    cohort: list[QASample],
    trajectories: list[AgentTrajectory],
    leakage: dict[str, int],
) -> None:
    completed_ids = {trajectory.example_id for trajectory in trajectories}
    completed_cohort = [sample for sample in cohort if sample.id in completed_ids]
    manifest = {
        "run_config": run_config,
        "completed": len(trajectories) == len(cohort),
        "cohort_examples": len(cohort),
        "cohort_ids_sha256": hashlib.sha256(
            ("\n".join(sample.id for sample in cohort) + "\n").encode("utf-8")
        ).hexdigest(),
        "leakage_check": leakage,
        "metrics": summarize(completed_cohort, trajectories) if trajectories else None,
    }
    temporary = path.with_name(path.name + ".building")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--retriever-url",
        default=os.environ.get("MAPD_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"),
    )
    parser.add_argument("--retriever-backend", required=True)
    parser.add_argument("--retriever-index", type=Path, required=True)
    parser.add_argument("--retriever-nprobe", type=int)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    all_samples = read_jsonl(args.qa, QASample)
    cohort = select_cohort(all_samples, count=args.count, seed=args.seed)
    leakage = assert_heldout(cohort, args.training_data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort.jsonl"
    rollout_path = args.output_dir / "rollouts.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    run_config = {
        "schema": RUN_SCHEMA,
        "model": _model_identity(args.model),
        "qa": str(args.qa.resolve()),
        "qa_sha256": _sha256(args.qa),
        "training_data": str(args.training_data.resolve()) if args.training_data else None,
        "training_data_sha256": _sha256(args.training_data) if args.training_data else None,
        "retriever_url": args.retriever_url,
        "retriever_backend": args.retriever_backend,
        "retriever_index": str(args.retriever_index.resolve()),
        "retriever_index_bytes": args.retriever_index.stat().st_size,
        "retriever_index_metadata_sha256": (
            _sha256(Path(str(args.retriever_index) + ".json"))
            if Path(str(args.retriever_index) + ".json").is_file()
            else None
        ),
        "retriever_nprobe": args.retriever_nprobe,
        "count": args.count,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_turns": args.max_turns,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt_sha256": hashlib.sha256(AGENT_SYSTEM_PROMPT.encode()).hexdigest(),
        "public_context_only": True,
    }

    completed: dict[str, AgentTrajectory] = {}
    if manifest_path.exists() or rollout_path.exists() or cohort_path.exists():
        if not (manifest_path.is_file() and rollout_path.is_file() and cohort_path.is_file()):
            raise RuntimeError("evaluation checkpoint is incomplete; use a new output directory")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("run_config") != run_config:
            raise RuntimeError("evaluation configuration changed; use a new output directory")
        prior_cohort = read_jsonl(cohort_path, QASample)
        if [item.id for item in prior_cohort] != [item.id for item in cohort]:
            raise RuntimeError("saved evaluation cohort changed")
        for trajectory in read_jsonl(rollout_path, AgentTrajectory):
            if trajectory.example_id in completed:
                raise ValueError("evaluation contains duplicate trajectories")
            completed[trajectory.example_id] = trajectory
        if not set(completed) <= {item.id for item in cohort}:
            raise ValueError("evaluation contains an unexpected example")
    else:
        write_jsonl(cohort_path, cohort)
        write_jsonl(rollout_path, [])
        _write_manifest(manifest_path, run_config, cohort, [], leakage)

    pending = [sample for sample in cohort if sample.id not in completed]
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

    for sample in pending:
        assert policy is not None and environment is not None
        trajectory = environment.rollout(
            sample, policy, max_new_tokens=args.max_new_tokens
        )
        completed[sample.id] = trajectory
        ordered = [completed[item.id] for item in cohort if item.id in completed]
        write_jsonl(rollout_path, ordered)
        _write_manifest(manifest_path, run_config, cohort, ordered, leakage)
        print(
            json.dumps(
                {
                    "event": "student_eval_progress",
                    "processed": len(ordered),
                    "total": len(cohort),
                    "example_id": sample.id,
                    "reward": trajectory.reward,
                    "answer": trajectory.final_answer,
                    "search_depth": sum(
                        turn.action == "search" for turn in trajectory.turns
                    ),
                    "invalid_actions": sum(
                        turn.action == "invalid" for turn in trajectory.turns
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ordered = [completed[item.id] for item in cohort]
    _write_manifest(manifest_path, run_config, cohort, ordered, leakage)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("STUDENT EVALUATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
