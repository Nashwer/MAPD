#!/usr/bin/env python3
"""Download the pinned smoke model into a deterministic local directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_ID = os.environ.get("MAPD_MODEL_ID", "Qwen/Qwen3-1.7B")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: download_model.py MODEL_DIRECTORY", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    output = Path(sys.argv[1]).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {output}", flush=True)
    resolved = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=output,
        max_workers=4,
    )
    print(f"Model ready: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
