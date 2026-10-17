#!/usr/bin/env python3
"""Create a paired baseline-versus-trained student evaluation report."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from mapd.environment.schema import AgentTrajectory
from mapd.io import read_jsonl


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def paired_report(
    baseline: list[AgentTrajectory],
    trained: list[AgentTrajectory],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    baseline_by_id = {item.example_id: item for item in baseline}
    trained_by_id = {item.example_id: item for item in trained}
    if len(baseline_by_id) != len(baseline) or len(trained_by_id) != len(trained):
        raise ValueError("evaluation files contain duplicate example IDs")
    if set(baseline_by_id) != set(trained_by_id):
        raise ValueError("baseline and trained evaluations use different cohorts")
    ids = sorted(baseline_by_id)
    if not ids:
        raise ValueError("evaluation cohort is empty")
    for example_id in ids:
        if baseline_by_id[example_id].prompt != trained_by_id[example_id].prompt:
            raise ValueError(f"prompt mismatch for {example_id}")

    baseline_rewards = [baseline_by_id[key].reward for key in ids]
    trained_rewards = [trained_by_id[key].reward for key in ids]
    differences = [after - before for before, after in zip(baseline_rewards, trained_rewards)]
    rng = random.Random(seed)
    bootstrapped = [
        mean(rng.choice(differences) for _ in differences)
        for _ in range(bootstrap_samples)
    ]

    def depth(item: AgentTrajectory) -> int:
        return sum(turn.action == "search" for turn in item.turns)

    baseline_depths = [depth(baseline_by_id[key]) for key in ids]
    trained_depths = [depth(trained_by_id[key]) for key in ids]
    both_correct = sum(before == 1 and after == 1 for before, after in zip(baseline_rewards, trained_rewards))
    baseline_only = sum(before == 1 and after == 0 for before, after in zip(baseline_rewards, trained_rewards))
    trained_only = sum(before == 0 and after == 1 for before, after in zip(baseline_rewards, trained_rewards))
    both_wrong = len(ids) - both_correct - baseline_only - trained_only
    return {
        "schema": "mapd-paired-student-eval-v1",
        "examples": len(ids),
        "baseline_accuracy": mean(baseline_rewards),
        "trained_accuracy": mean(trained_rewards),
        "accuracy_delta": mean(differences),
        "accuracy_delta_95pct_paired_bootstrap": [
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ],
        "paired_outcomes": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "trained_only_correct": trained_only,
            "both_wrong": both_wrong,
        },
        "baseline_mean_search_depth": mean(baseline_depths),
        "trained_mean_search_depth": mean(trained_depths),
        "mean_search_depth_delta": mean(
            after - before for before, after in zip(baseline_depths, trained_depths)
        ),
        "baseline_multi_search_rate": sum(value >= 2 for value in baseline_depths) / len(ids),
        "trained_multi_search_rate": sum(value >= 2 for value in trained_depths) / len(ids),
        "baseline_invalid_trajectory_rate": sum(
            any(turn.action == "invalid" for turn in baseline_by_id[key].turns) for key in ids
        ) / len(ids),
        "trained_invalid_trajectory_rate": sum(
            any(turn.action == "invalid" for turn in trained_by_id[key].turns) for key in ids
        ) / len(ids),
    }


def _markdown(report: dict[str, Any]) -> str:
    low, high = report["accuracy_delta_95pct_paired_bootstrap"]
    paired = report["paired_outcomes"]
    return "\n".join(
        [
            "# MAPD student evaluation comparison",
            "",
            f"- Examples: {report['examples']}",
            f"- Baseline accuracy: {report['baseline_accuracy']:.3%}",
            f"- Trained accuracy: {report['trained_accuracy']:.3%}",
            f"- Accuracy delta: {report['accuracy_delta']:+.3%} (paired bootstrap 95% CI {low:+.3%} to {high:+.3%})",
            f"- Baseline/trained mean search depth: {report['baseline_mean_search_depth']:.3f} / {report['trained_mean_search_depth']:.3f}",
            f"- Baseline/trained multi-search rate: {report['baseline_multi_search_rate']:.3%} / {report['trained_multi_search_rate']:.3%}",
            f"- Baseline/trained invalid-trajectory rate: {report['baseline_invalid_trajectory_rate']:.3%} / {report['trained_invalid_trajectory_rate']:.3%}",
            "",
            "## Paired outcomes",
            "",
            f"- Both correct: {paired['both_correct']}",
            f"- Baseline only correct: {paired['baseline_only_correct']}",
            f"- Trained only correct: {paired['trained_only_correct']}",
            f"- Both wrong: {paired['both_wrong']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_manifest_path = args.baseline.with_name("manifest.json")
    trained_manifest_path = args.trained.with_name("manifest.json")
    if not baseline_manifest_path.is_file() or not trained_manifest_path.is_file():
        raise FileNotFoundError("each rollout file must have a sibling manifest.json")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    trained_manifest = json.loads(trained_manifest_path.read_text(encoding="utf-8"))
    if not baseline_manifest.get("completed") or not trained_manifest.get("completed"):
        raise RuntimeError("both evaluations must be complete before comparison")
    if baseline_manifest.get("cohort_ids_sha256") != trained_manifest.get("cohort_ids_sha256"):
        raise ValueError("baseline and trained evaluation cohort hashes differ")
    baseline_config = dict(baseline_manifest.get("run_config", {}))
    trained_config = dict(trained_manifest.get("run_config", {}))
    baseline_config.pop("model", None)
    trained_config.pop("model", None)
    if baseline_config != trained_config:
        raise ValueError("baseline and trained evaluation settings differ")
    baseline = read_jsonl(args.baseline, AgentTrajectory)
    trained = read_jsonl(args.trained, AgentTrajectory)
    report = paired_report(baseline, trained)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "comparison.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PAIRED STUDENT EVALUATION COMPARISON OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
