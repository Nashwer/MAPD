#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mapd.retrieval.e5_ivfpq import build_e5_ivfpq_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a resumable low-memory E5 IVF-PQ wiki-18 index."
    )
    parser.add_argument(
        "--docstore",
        type=Path,
        default=Path("data/wiki18/index/wiki18.sqlite3"),
        help="Existing SQLite wiki-18 index; its passages table is reused as the document store.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/wiki18/e5_ivfpq/wiki18_e5_ivfpq.faiss"),
    )
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--passage-max-length", type=int, default=180)
    parser.add_argument("--nlist", type=int, default=4_096)
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=262_144)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument(
        "--limit",
        type=int,
        help="Build only the first N documents for a deployment smoke test.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_e5_ivfpq_index(
        args.docstore,
        args.index,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        passage_max_length=args.passage_max_length,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
        train_size=args.train_size,
        nprobe=args.nprobe,
        checkpoint_every=args.checkpoint_every,
        threads=args.threads,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("COMPRESSED E5 INDEX BUILD OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
