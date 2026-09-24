#!/usr/bin/env python3
"""Validate the Search-R1 HTTP contract and answer recall in real top-k results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mapd.data.schema import QASample
from mapd.io import read_jsonl
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.reward.exact_match import normalize_answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args()

    samples = read_jsonl(args.qa, QASample)[: args.limit]
    if not samples:
        raise ValueError("QA input is empty")
    retriever = HTTPRetriever(args.url, timeout_seconds=60)
    hits = 0
    examples = []
    for index, sample in enumerate(samples, start=1):
        print(f"retrieval query {index}/{len(samples)}: {sample.question}", flush=True)
        passages = retriever.search(sample.question, args.top_k)
        haystack = normalize_answer("\n".join(item.contents for item in passages))
        matched = any(
            normalized and re.search(rf"(?:^|\s){re.escape(normalized)}(?:$|\s)", haystack)
            for answer in sample.answers
            if (normalized := normalize_answer(answer))
        )
        hits += int(matched)
        examples.append(
            {
                "id": sample.id,
                "hit": matched,
                "retrieved": len(passages),
                "top_ids": [item.id for item in passages],
            }
        )
        print(
            f"retrieval result {index}/{len(samples)}: "
            f"documents={len(passages)} hit={matched}",
            flush=True,
        )
    recall = hits / len(samples)
    summary = {
        "queries": len(samples),
        "top_k": args.top_k,
        "answer_recall": recall,
        "examples": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if recall < args.min_recall:
        raise RuntimeError(f"answer recall {recall:.3f} is below {args.min_recall:.3f}")
    print("REAL RETRIEVAL SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
