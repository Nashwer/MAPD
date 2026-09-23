#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERL_VENV=${VERL_VENV:-"$(dirname "$PROJECT_ROOT")/verl/.venv"}
PYTHON="$VERL_VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON" >&2
  echo "Set VERL_VENV to the existing veRL virtual environment." >&2
  exit 1
fi

"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT"
if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON" -m pip install "pytest>=8,<9"
fi
cd "$PROJECT_ROOT"
"$PYTHON" -m mapd doctor --config configs/local_smoke.yaml --require-gpu
