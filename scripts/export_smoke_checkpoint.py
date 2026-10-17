#!/usr/bin/env python3
"""Merge a MAPD single-device smoke checkpoint into a loadable HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("trainable_state"), dict):
        raise ValueError("not a MAPD single-device checkpoint")
    return payload


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint).resolve()
    output = Path(output_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = _load_checkpoint(checkpoint_path)
    source_model = Path(base_model or payload.get("base_model", "")).resolve()
    if not (source_model / "config.json").is_file():
        raise FileNotFoundError(f"base model config not found: {source_model / 'config.json'}")

    checkpoint_sha256 = _sha256(checkpoint_path)
    existing_manifest = output / "mapd-export-manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("base_model") == str(source_model)
            and (output / "config.json").is_file()
        ):
            return {**existing, "reused": True}
        raise RuntimeError(f"output directory contains a different export: {output}")
    if output.exists():
        raise RuntimeError(f"incomplete export directory exists: {output}")

    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise RuntimeError(f"incomplete staging directory exists: {staging}")
    staging.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(source_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    current = model.state_dict()
    updates = payload["trainable_state"]
    unknown = sorted(set(updates) - set(current))
    mismatched = sorted(
        name
        for name, value in updates.items()
        if name in current and tuple(value.shape) != tuple(current[name].shape)
    )
    if unknown or mismatched:
        raise ValueError(
            f"checkpoint does not match base model: unknown={unknown}, shape_mismatch={mismatched}"
        )
    current.update(updates)
    model.load_state_dict(current, strict=True)
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(staging)

    manifest = {
        "schema": "mapd-merged-smoke-checkpoint-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "base_model": str(source_model),
        "training_step": int(payload.get("step", 0)),
        "updated_tensors": len(updates),
        "dtype": "bfloat16",
        "output": str(output),
        "reused": False,
    }
    (staging / "mapd-export-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    args = parser.parse_args()
    manifest = export_checkpoint(
        args.checkpoint, args.output_dir, base_model=args.base_model
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("MAPD CHECKPOINT EXPORT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
