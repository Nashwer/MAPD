from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mapd.cli import _run_synthesis
from mapd.config import load_config
from mapd.data.schema import QASample
from mapd.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MAPD protocols with the offline MAS")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every is not None and args.progress_every < 1:
        parser.error("--progress-every must be positive")
    config = load_config(args.config)
    if args.progress_every is not None:
        config.run.progress_every = args.progress_every
    samples = read_jsonl(args.input, QASample)
    samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]
    artifacts = _run_synthesis(
        config,
        samples,
        args.output,
    )
    passed = sum(artifact.quality.passed for artifact in artifacts)
    task_types = Counter(artifact.exploration.task_type.value for artifact in artifacts)
    exploration_success = sum(artifact.exploration.success for artifact in artifacts)
    findings = [
        finding for artifact in artifacts for finding in artifact.exploration.findings
    ]
    summary = {
        "generated": len(artifacts),
        "quality_passed": passed,
        "exploration_success": exploration_success,
        "task_types": dict(sorted(task_types.items())),
        "findings": len(findings),
        "nonempty_findings": sum(bool(finding.summary.strip()) for finding in findings),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if artifacts and len(task_types) == 1 and task_types.get("others") == len(artifacts):
        raise RuntimeError(
            "MAS classified every sample as others; inspect the orchestrator contract before training"
        )


if __name__ == "__main__":
    main()
