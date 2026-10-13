#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mapd.retrieval.http_service import serve_http_retriever
from mapd.retrieval.sqlite_fts import SQLiteFTSRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sqlite-fts5", "e5-ivfpq"), default="sqlite-fts5")
    parser.add_argument("--index", type=Path, default=Path("data/wiki18/index/wiki18.sqlite3"))
    parser.add_argument("--docstore", type=Path)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.backend == "sqlite-fts5":
        retriever = SQLiteFTSRetriever(
            args.index, query_timeout_seconds=args.query_timeout
        )
    else:
        from mapd.retrieval.e5_ivfpq import CompressedE5Retriever

        retriever = CompressedE5Retriever(
            args.index,
            args.docstore,
            model_name=args.model,
            device=args.device,
            nprobe=args.nprobe,
            threads=args.threads,
        )
    serve_http_retriever(
        retriever,
        host=args.host,
        port=args.port,
        default_top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
