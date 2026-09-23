#!/usr/bin/env python3
"""Write a compact machine-readable manifest for an ephemeral server instance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: capture_environment.py VERL_ROOT MODEL_ROOT", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    verl_root = Path(sys.argv[1]).resolve()
    model_root = Path(sys.argv[2]).expanduser().resolve()

    import torch

    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "schema": os.environ.get("MAPD_BOOTSTRAP_SCHEMA"),
            "verl_ref": os.environ.get("MAPD_VERL_REF"),
            "model_id": os.environ.get("MAPD_MODEL_ID"),
            "numpy": os.environ.get("MAPD_NUMPY_VERSION"),
            "cuda_series": os.environ.get("MAPD_CUDA_SERIES"),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("verl", "torch", "vllm", "numpy", "flashinfer-python", "pyarrow")
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvcc": command(["nvcc", "--version"]),
        },
        "git": {
            "mapd": command(["git", "rev-parse", "HEAD"], project_root),
            "verl": command(["git", "rev-parse", "HEAD"], verl_root),
        },
        "paths": {
            "project": str(project_root),
            "verl": str(verl_root),
            "model": str(model_root),
            "model_ready": (model_root / "config.json").is_file(),
        },
    }
    output = project_root / "artifacts" / "environment" / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Environment manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
