#!/usr/bin/env python3
"""Download versioned Search-R1 QA data and wiki-18 assets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


ASSETS = {
    "qa": (
        "PeterJinGo/nq_hotpotqa_train",
        ("train.parquet", "test.parquet"),
        "MAPD_QA_REVISION",
    ),
    "wiki": (
        "PeterJinGo/wiki-18-corpus",
        ("wiki-18.jsonl.gz",),
        "MAPD_WIKI18_REVISION",
    ),
    "e5-index": (
        "PeterJinGo/wiki-18-e5-index",
        ("part_aa", "part_ab"),
        "MAPD_WIKI18_E5_REVISION",
    ),
}
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=ASSETS)
    parser.add_argument("--output-root", type=Path, default=Path("data/downloads"))
    parser.add_argument("--revision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; run `bash mapd.sh bootstrap` first") from exc

    repo_id, filenames, revision_variable = ASSETS[args.asset]
    requested_revision = args.revision or os.environ.get(revision_variable) or "main"
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    # Repository defaults are immutable 40-character commit SHAs, so avoid a
    # separate metadata request on bandwidth-constrained servers.
    if _COMMIT_PATTERN.fullmatch(requested_revision):
        resolved_revision = requested_revision
    else:
        resolved_revision = _retry(
            lambda: HfApi(endpoint=endpoint)
            .dataset_info(repo_id, revision=requested_revision)
            .sha,
            description=f"resolve {repo_id}@{requested_revision}",
        )
    destination = args.output_root / args.asset
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        print(f"downloading {repo_id}@{resolved_revision}:{filename}", flush=True)
        paths.append(
            _retry(
                lambda filename=filename: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=resolved_revision,
                local_dir=destination,
                endpoint=endpoint,
                ),
                description=f"download {filename}",
            )
        )
    manifest = {
        "schema": "mapd-download-v1",
        "asset": args.asset,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "endpoint": endpoint,
        "files": paths,
    }
    (destination / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _retry(operation, *, description: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            delay = 2 ** attempt
            print(
                f"{description} failed ({attempt}/{attempts}); retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
