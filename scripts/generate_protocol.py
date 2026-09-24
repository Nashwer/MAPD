from __future__ import annotations

import argparse
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
    print(f"generated={len(artifacts)} passed={passed} output={args.output}")


if __name__ == "__main__":
    main()
