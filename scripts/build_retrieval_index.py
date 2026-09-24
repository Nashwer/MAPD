#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapd.retrieval.sqlite_fts import SQLiteFTSRetriever, build_sqlite_fts_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("data/wiki18/index/wiki18.sqlite3"))
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--source-revision")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.index.is_file() and not args.force:
        retriever = SQLiteFTSRetriever(args.index)
        existing = retriever.metadata()
        if existing.get("source_revision") == args.source_revision:
            print(json.dumps({**existing, "index": str(args.index.resolve())}, indent=2))
            print("WIKI18 INDEX REUSED")
            return 0
    result = build_sqlite_fts_index(
        args.corpus,
        args.index,
        batch_size=args.batch_size,
        source_revision=args.source_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("WIKI18 INDEX BUILD OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
