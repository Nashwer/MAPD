#!/usr/bin/env python3
"""Validate and export the deterministic MAPD QA training reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapd.data.schema import QASample


BUNDLE_SCHEMA = "mapd-training-data-bundle-v1"
DATASET_SCHEMA = "mapd-data-v1"
EXPECTED_SOURCE = "PeterJinGo/nq_hotpotqa_train"
EXPECTED_TARGET_SIZE = 25_600
EXPECTED_SEED = 42


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    member.mode = 0o600
    archive.addfile(member, io.BytesIO(payload))


def _validate_training_jsonl(path: Path) -> tuple[list[str], Counter[str]]:
    ids: list[str] = []
    sources: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = QASample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid QA record at {path}:{line_number}") from exc
            if sample.id in seen:
                raise ValueError(f"duplicate sample id in training set: {sample.id}")
            seen.add(sample.id)
            ids.append(sample.id)
            sources[sample.data_source] += 1
    if len(ids) != EXPECTED_TARGET_SIZE:
        raise ValueError(
            f"expected {EXPECTED_TARGET_SIZE} training records, found {len(ids)} in {path}"
        )
    return ids, sources


def export_data_bundle(
    data_root: str | Path,
    output_path: str | Path,
    *,
    expected_revision: str | None = None,
    include_source: bool = False,
) -> dict[str, Any]:
    root = Path(data_root)
    training_path = root / "training" / "mapd_train_25600.jsonl"
    training_manifest_path = root / "training" / "manifest.json"
    heldout_path = root / "evaluation" / "searchr1_test.jsonl"
    for required in (training_path, training_manifest_path, heldout_path):
        if not required.is_file():
            raise FileNotFoundError(f"required prepared data file is missing: {required}")

    source_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    required_metadata = {
        "schema": DATASET_SCHEMA,
        "source": EXPECTED_SOURCE,
        "target_size": EXPECTED_TARGET_SIZE,
        "seed": EXPECTED_SEED,
    }
    for key, expected in required_metadata.items():
        if source_manifest.get(key) != expected:
            raise ValueError(
                f"unexpected training manifest {key}: "
                f"expected {expected!r}, got {source_manifest.get(key)!r}"
            )
    revision = source_manifest.get("source_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("training manifest has no pinned source_revision")
    if expected_revision and revision != expected_revision:
        raise ValueError(
            f"source revision mismatch: expected {expected_revision}, got {revision}"
        )

    sample_ids, selected_counts = _validate_training_jsonl(training_path)
    manifest_counts = Counter(source_manifest.get("selected_counts") or {})
    if selected_counts != manifest_counts:
        raise ValueError(
            f"selected source counts do not match manifest: {dict(selected_counts)} "
            f"!= {dict(manifest_counts)}"
        )

    relative_files = [
        Path("training/mapd_train_25600.jsonl"),
        Path("training/manifest.json"),
        Path("evaluation/searchr1_test.jsonl"),
    ]
    download_manifest = root / "downloads" / "qa" / "download-manifest.json"
    if download_manifest.is_file():
        relative_files.append(Path("downloads/qa/download-manifest.json"))
    if include_source:
        for name in ("train.parquet", "test.parquet"):
            relative = Path("downloads/qa") / name
            if not (root / relative).is_file():
                raise FileNotFoundError(f"raw source file is missing: {root / relative}")
            relative_files.append(relative)

    payloads = {relative.as_posix(): (root / relative).read_bytes() for relative in relative_files}
    note = (
        "This bundle contains the repository's deterministic 25,600-example "
        "reconstruction (12,800 NQ + 12,800 HotpotQA, seed 42), not the paper "
        "authors' unpublished exact sample-ID list. The wiki-18 retrieval corpus, "
        "generated teacher protocols, and the paper's seven evaluation datasets "
        "are intentionally not included.\n"
    ).encode("utf-8")
    payloads["README.txt"] = note

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reproduction_status": "deterministic-reconstruction; exact paper sample IDs unpublished",
        "source": EXPECTED_SOURCE,
        "source_revision": revision,
        "seed": EXPECTED_SEED,
        "training_examples": len(sample_ids),
        "selected_counts": dict(sorted(selected_counts.items())),
        "sample_ids_sha256": _sha256(("\n".join(sample_ids) + "\n").encode("utf-8")),
        "includes_raw_source_parquet": include_source,
        "files": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(payloads.items())
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz") as archive:
        _add_bytes(archive, "bundle-manifest.json", _json_bytes(manifest))
        for name, payload in sorted(payloads.items()):
            _add_bytes(archive, f"payload/data/{name}", payload)
    os.replace(temporary, destination)
    return {
        **manifest,
        "bundle": str(destination.resolve()),
        "bundle_sha256": _sha256(destination.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--include-source", action="store_true")
    args = parser.parse_args()
    result = export_data_bundle(
        args.data_root,
        args.output,
        expected_revision=args.expected_revision,
        include_source=args.include_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
