#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_VENV=${VERL_VENV:-"$(dirname "$PROJECT_ROOT")/verl/.venv"}

if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
  export PATH="$CUDA_HOME/bin:$PATH"
fi

usage() {
  cat <<'EOF'
Usage: bash mapd.sh COMMAND

Commands:
  setup    Install into the sibling veRL environment and run full verification
  verify   Run tests and the offline end-to-end smoke flow
  model-smoke  Load the local Qwen model and run one GPU inference
  agent-smoke  Run one real Qwen -> BM25 search -> answer trajectory
  start    Run verification in the background
  status   Show background job and GPU status
  logs     Show the last 200 log lines
  follow   Follow logs; Ctrl-C leaves the job running
  stop     Gracefully stop the background job
  jobs     List known background jobs
  doctor   Check Python, veRL, vLLM, and CUDA
EOF
}

case ${1:-} in
  setup)
    bash "$PROJECT_ROOT/scripts/install_linux.sh"
    bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    ;;
  verify)
    bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    ;;
  model-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/model_smoke.py" "${2:-}"
    ;;
  agent-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/agent_smoke.py" "${2:-}"
    ;;
  start)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start verify bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    ;;
  status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status verify
    "$VERL_VENV/bin/python" -m mapd status --artifact-dir "$PROJECT_ROOT/artifacts/smoke"
    ;;
  logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs verify 200
    ;;
  follow)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" follow verify
    ;;
  stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop verify
    ;;
  jobs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" list
    ;;
  doctor)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" -m mapd doctor --config configs/local_smoke.yaml --require-gpu
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
