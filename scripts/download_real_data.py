#!/usr/bin/env python3
"""Download versioned Search-R1 QA data and wiki-18 assets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
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
    resolved_revision = HfApi().dataset_info(repo_id, revision=requested_revision).sha
    destination = args.output_root / args.asset
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        print(f"downloading {repo_id}@{resolved_revision}:{filename}", flush=True)
        paths.append(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=resolved_revision,
                local_dir=destination,
            )
        )
    manifest = {
        "schema": "mapd-download-v1",
        "asset": args.asset,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "files": paths,
    }
    (destination / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
