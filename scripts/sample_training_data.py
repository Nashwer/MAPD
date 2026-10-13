#!/usr/bin/env python3
"""Create a deterministic, source-stratified subset of prepared MAPD QA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapd.data.schema import QASample
from mapd.io import read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, records: list[QASample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    os.replace(temporary, path)


def _allocate_quotas(counts: Counter[str], target_size: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {source: target_size * count / total for source, count in counts.items()}
    quotas = {source: math.floor(value) for source, value in exact.items()}
    remaining = target_size - sum(quotas.values())
    order = sorted(counts, key=lambda source: (-(exact[source] - quotas[source]), source))
    for source in order[:remaining]:
        quotas[source] += 1
    return quotas


def sample_training_subset(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    source = Path(input_path)
    samples = read_jsonl(source, QASample)
    if not samples:
        raise ValueError("input training data is empty")
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("input training data contains duplicate sample IDs")

    grouped: dict[str, list[QASample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.data_source].append(sample)
    input_counts = Counter({key: len(value) for key, value in grouped.items()})
    target_size = round(len(samples) * fraction)
    if target_size < 1:
        raise ValueError("fraction selects fewer than one record")
    quotas = _allocate_quotas(input_counts, target_size)

    rng = random.Random(seed)
    selected: list[QASample] = []
    for data_source in sorted(grouped):
        candidates = list(grouped[data_source])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[data_source]])
    rng.shuffle(selected)

    output = Path(output_path)
    manifest_output = Path(manifest_path)
    _write_jsonl_atomic(output, selected)
    selected_counts = Counter(sample.data_source for sample in selected)
    manifest = {
        "schema": "mapd-training-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic source-stratified random sample without replacement",
        "seed": seed,
        "fraction": fraction,
        "input": str(source),
        "input_examples": len(samples),
        "input_counts": dict(sorted(input_counts.items())),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_examples": len(selected),
        "selected_counts": dict(sorted(selected_counts.items())),
        "output_sha256": _sha256(output),
        "selected_ids_sha256": hashlib.sha256(
            ("\n".join(sample.id for sample in selected) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_output.with_name(manifest_output.name + ".building")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--percent", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = sample_training_subset(
        args.input,
        args.output,
        args.manifest,
        fraction=args.percent / 100.0,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
