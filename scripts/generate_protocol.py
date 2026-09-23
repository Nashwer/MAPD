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
    args = parser.parse_args()
    artifacts = _run_synthesis(
        load_config(args.config),
        read_jsonl(args.input, QASample),
        args.output,
    )
    passed = sum(artifact.quality.passed for artifact in artifacts)
    print(f"generated={len(artifacts)} passed={passed} output={args.output}")


if __name__ == "__main__":
    main()

