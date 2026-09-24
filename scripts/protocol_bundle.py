#!/usr/bin/env python3
"""Export, restore, and merge portable MAPD protocol artifact bundles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mapd.data.verl import to_verl_record
from mapd.io import read_jsonl, stable_hash
from mapd.mas.schema import SynthesisArtifact


BUNDLE_SCHEMA = "mapd-protocol-bundle-v1"
PAYLOAD_NAMES = (
    "artifacts.jsonl",
    "explorations.jsonl",
    "protocols.jsonl",
    "quality.jsonl",
    "training.jsonl",
    "manifest.json",
    "teacher_usage.jsonl",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(records: Iterable[Any]) -> bytes:
    lines = []
    for record in records:
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_source(source: Path, project_root: Path) -> str:
    try:
        relative = source.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("protocol artifacts must be inside the repository") from exc
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ValueError("protocol artifacts must be under artifacts/")
    return relative.as_posix()


def export_protocol_bundle(
    source_dir: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path,
    dataset_offset: int,
    requested_count: int,
) -> dict[str, Any]:
    source = Path(source_dir)
    artifacts_path = source / "artifacts.jsonl"
    if not artifacts_path.is_file():
        raise FileNotFoundError(f"no completed protocol checkpoint: {artifacts_path}")
    artifacts = read_jsonl(artifacts_path, SynthesisArtifact)
    if not artifacts:
        raise ValueError("protocol checkpoint is empty")

    payloads: dict[str, bytes] = {}
    for name in PAYLOAD_NAMES:
        path = source / name
        if path.is_file():
            payloads[name] = (
                _normalized_usage_bytes(path)
                if name == "teacher_usage.jsonl"
                else path.read_bytes()
            )
    # artifacts.jsonl is always normalized and validated before export.
    payloads["artifacts.jsonl"] = _jsonl_bytes(artifacts)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact_dir": _safe_relative_source(source, Path(project_root)),
        "dataset_offset": dataset_offset,
        "requested_count": requested_count,
        "completed_count": len(artifacts),
        "passed_count": sum(item.quality.passed for item in artifacts),
        "sample_ids": [item.sample.id for item in artifacts],
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
            _add_bytes(archive, f"payload/{name}", payload)
    os.replace(temporary, destination)
    return {
        **manifest,
        "bundle": str(destination.resolve()),
        "bundle_sha256": _sha256(destination.read_bytes()),
    }


def restore_protocol_bundle(
    bundle_path: str | Path, *, project_root: str | Path
) -> dict[str, Any]:
    manifest, payloads = _read_bundle(Path(bundle_path))
    relative = Path(str(manifest["source_artifact_dir"]))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "artifacts":
        raise ValueError("bundle contains an unsafe artifact destination")
    destination = Path(project_root) / relative
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        target = destination / name
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite different restored artifact: {target}"
            )
        target.write_bytes(payload)
    usage_payload = payloads.get("teacher_usage.jsonl")
    if usage_payload:
        _merge_usage_ledger(
            Path(project_root) / "artifacts/teacher_usage.jsonl", usage_payload
        )
    return {
        "bundle": str(Path(bundle_path).resolve()),
        "restored_to": str(destination.resolve()),
        "completed_count": manifest["completed_count"],
    }


def merge_protocol_bundles(
    bundle_paths: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    loaded = []
    for bundle_path in bundle_paths:
        path = Path(bundle_path)
        manifest, payloads = _read_bundle(path)
        artifacts = _artifacts_from_bytes(payloads["artifacts.jsonl"], path)
        loaded.append((int(manifest["dataset_offset"]), path, manifest, payloads, artifacts))
    if not loaded:
        raise ValueError("at least one protocol bundle is required")

    merged: list[SynthesisArtifact] = []
    by_id: dict[str, SynthesisArtifact] = {}
    usage_records: list[dict[str, Any]] = []
    usage_seen: set[str] = set()
    for _, _, _, payloads, artifacts in sorted(loaded, key=lambda item: (item[0], str(item[1]))):
        for artifact in artifacts:
            existing = by_id.get(artifact.sample.id)
            if existing is not None:
                if stable_hash(existing) != stable_hash(artifact):
                    raise ValueError(f"conflicting protocol for sample {artifact.sample.id}")
                continue
            by_id[artifact.sample.id] = artifact
            merged.append(artifact)
        usage = payloads.get("teacher_usage.jsonl", b"").decode("utf-8")
        for line in usage.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = stable_hash(record)
            if key not in usage_seen:
                usage_seen.add(key)
                usage_records.append(record)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_derived_artifacts(destination, merged)
    if usage_records:
        (destination / "teacher_usage.jsonl").write_bytes(_jsonl_bytes(usage_records))
    result = {
        "schema": "mapd-protocol-merge-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(merged),
        "passed": sum(item.quality.passed for item in merged),
        "bundle_count": len(loaded),
        "bundles": [str(item[1]) for item in loaded],
        "estimated_cost_usd": round(
            sum(float(item.get("estimated_cost_usd") or 0) for item in usage_records), 8
        ),
    }
    (destination / "manifest.json").write_bytes(_json_bytes(result))
    return {**result, "output": str(destination.resolve())}


def _write_derived_artifacts(
    destination: Path, artifacts: list[SynthesisArtifact]
) -> None:
    (destination / "artifacts.jsonl").write_bytes(_jsonl_bytes(artifacts))
    (destination / "explorations.jsonl").write_bytes(
        _jsonl_bytes(item.exploration for item in artifacts)
    )
    (destination / "protocols.jsonl").write_bytes(
        _jsonl_bytes(
            item.protocol if item.protocol is not None else item.raw_protocol
            for item in artifacts
        )
    )
    (destination / "quality.jsonl").write_bytes(
        _jsonl_bytes(item.quality for item in artifacts)
    )
    (destination / "training.jsonl").write_bytes(
        _jsonl_bytes(to_verl_record(item, index) for index, item in enumerate(artifacts))
    )


def _read_bundle(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.getmember("bundle-manifest.json")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise ValueError("bundle manifest is unreadable")
        manifest = json.load(manifest_file)
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported protocol bundle schema: {manifest.get('schema')}")
        payloads: dict[str, bytes] = {}
        for name, expected in manifest["files"].items():
            if name not in PAYLOAD_NAMES:
                raise ValueError(f"unexpected bundle payload: {name}")
            member = archive.getmember(f"payload/{name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable bundle payload: {name}")
            payload = handle.read()
            if len(payload) != int(expected["bytes"]) or _sha256(payload) != expected["sha256"]:
                raise ValueError(f"checksum mismatch for bundle payload: {name}")
            payloads[name] = payload
    if "artifacts.jsonl" not in payloads:
        raise ValueError("bundle contains no artifacts.jsonl")
    return manifest, payloads


def _artifacts_from_bytes(payload: bytes, source: Path) -> list[SynthesisArtifact]:
    artifacts = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            artifacts.append(SynthesisArtifact.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid artifact in {source}:{line_number}") from exc
    return artifacts


def _normalized_usage_bytes(path: Path) -> bytes:
    """Ignore a final partial append if export races an in-flight API call."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return _jsonl_bytes(records)


def _merge_usage_ledger(path: Path, payload: bytes) -> None:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            seen.add(stable_hash(record))
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = stable_hash(record)
        if key not in seen:
            seen.add(key)
            records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    temporary.write_bytes(_jsonl_bytes(records))
    os.replace(temporary, path)


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    member.mode = 0o600
    archive.addfile(member, io.BytesIO(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    export_parser.add_argument("--offset", type=int, required=True)
    export_parser.add_argument("--count", type=int, required=True)
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("bundle", type=Path)
    restore_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("bundles", type=Path, nargs="+")
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        if args.offset < 0 or args.count < 1:
            parser.error("offset must be nonnegative and count must be positive")
        result = export_protocol_bundle(
            args.source,
            args.output,
            project_root=args.project_root,
            dataset_offset=args.offset,
            requested_count=args.count,
        )
    elif args.command == "restore":
        result = restore_protocol_bundle(args.bundle, project_root=args.project_root)
    else:
        result = merge_protocol_bundles(args.bundles, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
