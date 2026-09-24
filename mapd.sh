#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  # .env is ignored by Git and treated as trusted shell configuration.
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi
export VERL_VENV=${VERL_VENV:-"$(dirname "$PROJECT_ROOT")/verl/.venv"}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-"$HOME/cache/huggingface"}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1200}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
PROTOCOL_SYNTHESIS_REVISION=8
if [[ -f "$PROJECT_ROOT/scripts/bootstrap_versions.env" ]]; then
  # shellcheck source=scripts/bootstrap_versions.env
  source "$PROJECT_ROOT/scripts/bootstrap_versions.env"
  export MAPD_QA_REVISION MAPD_WIKI18_REVISION MAPD_WIKI18_E5_REVISION
fi

if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
  export PATH="$CUDA_HOME/bin:$PATH"
fi

usage() {
  cat <<'EOF'
Usage: bash mapd.sh COMMAND

Commands:
  bootstrap  Rebuild or repair the complete ephemeral GPU-server environment
  setup      Install into the sibling veRL environment and run full verification
  verify     Run tests and the offline end-to-end smoke flow
  data-setup Download and prepare the de-duplicated 25,600-example training set
  wiki-setup Download wiki-18 and build/reuse the persistent SQLite FTS5 index
  wiki-start/status/logs/stop  Manage wiki-18 setup as a background job
  retrieval-start/status/logs/stop  Manage the command-line retrieval service
  retrieval-smoke  Check real top-3 retrieval against normalized QA examples
  grpo-smoke Run real rollouts until rewards vary, then perform one GRPO update
  protocol-smoke  Generate a budget-capped real-teacher protocol sample
  protocol-shard OFFSET COUNT  Generate one deterministic resumable shard
  protocol-start/status/logs/stop  Manage a protocol shard background job
  protocol-export OFFSET COUNT  Create a checksummed portable shard bundle
  protocol-restore BUNDLE  Restore a bundle to its original artifact directory
  protocol-merge BUNDLE...  Merge and de-duplicate exported protocol bundles
  protocol-train-smoke OFFSET COUNT  Train on one real protocol shard and reload checkpoint
  protocol-train-start/status/logs/stop  Manage the real-protocol training smoke job
  paper-examples  Run Appendix A's three multi-hop examples through Qwen + wiki-18
  paper-examples-start/status/logs/stop  Manage the paper-example smoke job
  model-smoke  Load the local Qwen model and run one GPU inference
  agent-smoke  Run one real Qwen -> BM25 search -> answer trajectory
  train-smoke  Run rollout, dual-context MAPD update, and checkpoint reload
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
  bootstrap)
    bash "$PROJECT_ROOT/scripts/bootstrap_server.sh" "${@:2}"
    ;;
  setup)
    bash "$PROJECT_ROOT/scripts/install_linux.sh"
    bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    ;;
  verify)
    bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    ;;
  data-setup)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" scripts/download_real_data.py qa
    qa_revision=$("$VERL_VENV/bin/python" -c 'import json; print(json.load(open("data/downloads/qa/download-manifest.json"))["resolved_revision"])')
    "$VERL_VENV/bin/python" scripts/prepare_real_data.py \
      --train data/downloads/qa/train.parquet \
      --heldout data/downloads/qa/test.parquet \
      --source-revision "$qa_revision"
    ;;
  wiki-setup)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" scripts/download_real_data.py wiki
    wiki_revision=$("$VERL_VENV/bin/python" -c 'import json; print(json.load(open("data/downloads/wiki/download-manifest.json"))["resolved_revision"])')
    "$VERL_VENV/bin/python" scripts/build_retrieval_index.py \
      --corpus data/downloads/wiki/wiki-18.jsonl.gz \
      --index data/wiki18/index/wiki18.sqlite3 \
      --source-revision "$wiki_revision"
    ;;
  wiki-start)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start wiki-setup bash "$PROJECT_ROOT/mapd.sh" wiki-setup
    ;;
  wiki-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status wiki-setup
    ;;
  wiki-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs wiki-setup 100
    ;;
  wiki-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop wiki-setup
    ;;
  retrieval-start)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start retriever \
      "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/serve_retriever.py" \
      --index "$PROJECT_ROOT/data/wiki18/index/wiki18.sqlite3"
    for _ in {1..30}; do
      if curl -fsS --connect-timeout 1 --max-time 2 http://127.0.0.1:8000/health; then
        echo
        exit 0
      fi
      sleep 1
    done
    echo "retriever did not become healthy; inspect: bash mapd.sh retrieval-logs" >&2
    exit 1
    ;;
  retrieval-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status retriever
    ;;
  retrieval-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs retriever 100
    ;;
  retrieval-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop retriever
    ;;
  retrieval-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" scripts/retrieval_smoke.py \
      --qa data/training/mapd_train_25600.jsonl --limit "${2:-20}" --top-k 3
    ;;
  grpo-smoke)
    cd "$PROJECT_ROOT"
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    "$VERL_VENV/bin/python" scripts/real_grpo_rollout.py \
      --model "$MODEL_PATH" --qa data/training/mapd_train_25600.jsonl
    "$VERL_VENV/bin/python" scripts/real_grpo_optimize.py --model "$MODEL_PATH"
    ;;
  protocol-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" scripts/generate_protocol.py \
      --config configs/paper_like.yaml \
      --input data/training/mapd_train_25600.jsonl \
      --output "artifacts/protocol_smoke_r${PROTOCOL_SYNTHESIS_REVISION}" \
      --limit "${2:-20}" \
      --progress-every 1
    ;;
  protocol-shard)
    cd "$PROJECT_ROOT"
    OFFSET=${2:-0}
    COUNT=${3:-20}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_r${PROTOCOL_SYNTHESIS_REVISION}"
    "$VERL_VENV/bin/python" scripts/generate_protocol.py \
      --config configs/paper_like.yaml \
      --input data/training/mapd_train_25600.jsonl \
      --output "$SHARD_DIR" \
      --offset "$OFFSET" \
      --limit "$COUNT" \
      --progress-every 1
    ;;
  protocol-start)
    OFFSET=${2:-0}
    COUNT=${3:-20}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start protocol-shard \
      bash "$PROJECT_ROOT/mapd.sh" protocol-shard "$OFFSET" "$COUNT"
    ;;
  protocol-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status protocol-shard
    ;;
  protocol-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs protocol-shard 100
    ;;
  protocol-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop protocol-shard
    ;;
  protocol-export)
    cd "$PROJECT_ROOT"
    OFFSET=${2:-0}
    COUNT=${3:-20}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_r${PROTOCOL_SYNTHESIS_REVISION}"
    "$VERL_VENV/bin/python" scripts/protocol_bundle.py export \
      --source "$SHARD_DIR" \
      --output "$PROJECT_ROOT/exports/protocol_offset_${OFFSET}_count_${COUNT}_r${PROTOCOL_SYNTHESIS_REVISION}.tar.gz" \
      --project-root "$PROJECT_ROOT" \
      --offset "$OFFSET" \
      --count "$COUNT"
    ;;
  protocol-restore)
    cd "$PROJECT_ROOT"
    [[ -n "${2:-}" ]] || { echo "BUNDLE is required" >&2; exit 2; }
    "$VERL_VENV/bin/python" scripts/protocol_bundle.py restore \
      --project-root "$PROJECT_ROOT" "$2"
    ;;
  protocol-merge)
    cd "$PROJECT_ROOT"
    shift
    [[ "$#" -gt 0 ]] || { echo "at least one BUNDLE is required" >&2; exit 2; }
    "$VERL_VENV/bin/python" scripts/protocol_bundle.py merge \
      --output "$PROJECT_ROOT/artifacts/protocol_merged" "$@"
    ;;
  protocol-train-smoke)
    cd "$PROJECT_ROOT"
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_r${PROTOCOL_SYNTHESIS_REVISION}"
    [[ -s "$SHARD_DIR/artifacts.jsonl" ]] || {
      echo "protocol shard not found: $SHARD_DIR/artifacts.jsonl" >&2
      exit 1
    }
    if ! curl -fsS --connect-timeout 1 --max-time 3 http://127.0.0.1:8000/health >/dev/null; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    OUTPUT_DIR="$PROJECT_ROOT/artifacts/protocol_train_smoke/offset_${OFFSET}_count_${COUNT}_r${PROTOCOL_SYNTHESIS_REVISION}_action_v3"
    "$VERL_VENV/bin/python" scripts/protocol_train_smoke_rollout.py \
      --model "$MODEL_PATH" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --output-dir "$OUTPUT_DIR" \
      --retriever-url "${MAPD_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
    "$VERL_VENV/bin/python" scripts/protocol_train_smoke_optimize.py \
      --model "$MODEL_PATH" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --rollouts "$OUTPUT_DIR/rollouts.jsonl" \
      --output-dir "$OUTPUT_DIR"
    ;;
  protocol-train-start)
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start protocol-train-smoke \
      bash "$PROJECT_ROOT/mapd.sh" protocol-train-smoke "$OFFSET" "$COUNT" "$MODEL_PATH"
    ;;
  protocol-train-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status protocol-train-smoke
    ;;
  protocol-train-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs protocol-train-smoke 100
    ;;
  protocol-train-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop protocol-train-smoke
    ;;
  paper-examples)
    cd "$PROJECT_ROOT"
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    if ! curl -fsS --connect-timeout 1 --max-time 3 http://127.0.0.1:8000/health >/dev/null; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    "$VERL_VENV/bin/python" scripts/paper_examples_smoke.py \
      --model "$MODEL_PATH" \
      --retriever-url "${MAPD_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
    ;;
  paper-examples-start)
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start paper-examples \
      bash "$PROJECT_ROOT/mapd.sh" paper-examples "$MODEL_PATH"
    ;;
  paper-examples-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status paper-examples
    ;;
  paper-examples-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs paper-examples 200
    ;;
  paper-examples-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop paper-examples
    ;;
  model-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/model_smoke.py" "${2:-}"
    ;;
  agent-smoke)
    cd "$PROJECT_ROOT"
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/agent_smoke.py" "${2:-}"
    ;;
  train-smoke)
    cd "$PROJECT_ROOT"
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}}
    if [[ ! -s "$PROJECT_ROOT/artifacts/smoke/artifacts.jsonl" ]]; then
      bash "$PROJECT_ROOT/scripts/run_smoke.sh"
    fi
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/training_smoke_rollout.py" \
      --model "$MODEL_PATH"
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/training_smoke_optimize.py" \
      --model "$MODEL_PATH"
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
