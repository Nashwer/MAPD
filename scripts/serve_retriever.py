#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mapd.retrieval.http_service import serve_retriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/wiki18/index/wiki18.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query-timeout", type=float, default=20.0)
    args = parser.parse_args()
    serve_retriever(
        str(args.index),
        host=args.host,
        port=args.port,
        default_top_k=args.top_k,
        query_timeout_seconds=args.query_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
