#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERL_VENV=${VERL_VENV:-"$(dirname "$PROJECT_ROOT")/verl/.venv"}
PYTHON="$VERL_VENV/bin/python"

cd "$PROJECT_ROOT"
"$PYTHON" -m mapd doctor --config configs/local_smoke.yaml
"$PYTHON" -m pytest -q
"$PYTHON" -m mapd smoke --config configs/local_smoke.yaml
"$PYTHON" -m mapd validate --artifacts artifacts/smoke/artifacts.jsonl
"$PYTHON" -m mapd prepare-data \
  --artifacts artifacts/smoke/artifacts.jsonl \
  --output data/protocols/train_smoke.jsonl

