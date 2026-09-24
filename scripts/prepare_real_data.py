#!/usr/bin/env python3
"""Normalize, de-duplicate, isolate held-out questions, and select 25,600 examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapd.data.prepare import prepare_searchr1_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--eval-file", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--target-size", type=int, default=25_600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-revision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_searchr1_dataset(
        args.train,
        args.output_root,
        heldout_parquet=args.heldout,
        evaluation_jsonl=args.eval_file,
        target_size=args.target_size,
        seed=args.seed,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("REAL DATA PREP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
