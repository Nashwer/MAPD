#!/usr/bin/env python3
"""Create one review bundle with two teacher failures and one shallow student failure."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from export_shallow_trace import discover_rollouts, export_shallow_trace
from export_teacher_negative_trace import DEFAULT_EXAMPLE_IDS, export_teacher_negative


def export_review_bundle(
    artifacts: Path,
    rollouts: Path,
    output_root: Path,
) -> dict[str, object]:
    package = output_root / "retrieval_depth_review"
    package.mkdir(parents=True, exist_ok=True)
    teacher_outputs = [
        export_teacher_negative(artifacts, example_id, package)
        for example_id in DEFAULT_EXAMPLE_IDS
    ]
    student_output = export_shallow_trace(
        rollouts,
        package,
        example_id="mapd-paper-example-2",
        create_archive=False,
    )
    index = {
        "schema": "mapd-retrieval-depth-review-bundle-v1",
        "teacher_negatives": teacher_outputs,
        "student_negative": student_output,
        "selection": {
            "teacher": list(DEFAULT_EXAMPLE_IDS),
            "student": "terminated zero-reward rollout with the fewest search actions",
        },
    }
    (package / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package / "README.md").write_text(
        "# Retrieval-depth review package\n\n"
        "This package contains two DeepSeek/MAS teacher negative traces and one Qwen student "
        "negative trace. Each subdirectory has a concise `trace_readable.md` and the complete "
        "persisted `trace_complete.json`.\n\n"
        "Teacher traces diagnose effective dependency depth and query quality. The student trace "
        "diagnoses early stopping (`search -> unsupported answer`).\n",
        encoding="utf-8",
    )
    archive = output_root / "retrieval_depth_review.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(package, arcname=package.name)
    return {
        "package": str(package.resolve()),
        "archive": str(archive.resolve()),
        "teacher_examples": list(DEFAULT_EXAMPLE_IDS),
        "student_example": student_output["selected_example"],
        "student_search_depth": student_output["search_depth"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("exports"))
    args = parser.parse_args()
    rollouts = args.rollouts or discover_rollouts(args.project_root)
    print(
        json.dumps(
            export_review_bundle(args.artifacts, rollouts, args.output_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
