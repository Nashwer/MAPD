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
export MAPD_STORAGE_ROOT=${MAPD_STORAGE_ROOT:-"$HOME"}
export MAPD_CACHE_ROOT=${MAPD_CACHE_ROOT:-"$MAPD_STORAGE_ROOT/cache"}
export MAPD_MODEL_ROOT=${MAPD_MODEL_ROOT:-"$MAPD_STORAGE_ROOT/models"}
export VERL_ROOT=${VERL_ROOT:-"$(dirname "$PROJECT_ROOT")/verl"}
export VERL_VENV=${VERL_VENV:-"$VERL_ROOT/.venv"}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-"$MAPD_CACHE_ROOT/huggingface"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$MAPD_CACHE_ROOT/uv"}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$MAPD_CACHE_ROOT/pip"}
export TMPDIR=${TMPDIR:-"$MAPD_STORAGE_ROOT/tmp"}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1200}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
PROTOCOL_SYNTHESIS_REVISION=10
if [[ -f "$PROJECT_ROOT/scripts/bootstrap_versions.env" ]]; then
  # shellcheck source=scripts/bootstrap_versions.env
  source "$PROJECT_ROOT/scripts/bootstrap_versions.env"
  export MAPD_QA_REVISION MAPD_WIKI18_REVISION MAPD_WIKI18_E5_REVISION
fi
export MAPD_RETRIEVER_HOST=${MAPD_RETRIEVER_HOST:-127.0.0.1}
export MAPD_RETRIEVER_PORT=${MAPD_RETRIEVER_PORT:-8000}
[[ "$MAPD_RETRIEVER_PORT" =~ ^[0-9]+$ ]] &&
  (( MAPD_RETRIEVER_PORT >= 1 && MAPD_RETRIEVER_PORT <= 65535 )) || {
  echo "MAPD_RETRIEVER_PORT must be an integer between 1 and 65535" >&2
  exit 2
}
export MAPD_RETRIEVER_URL=${MAPD_RETRIEVER_URL:-"http://$MAPD_RETRIEVER_HOST:$MAPD_RETRIEVER_PORT/retrieve"}
export MAPD_RETRIEVER_HEALTH_URL=${MAPD_RETRIEVER_HEALTH_URL:-"${MAPD_RETRIEVER_URL%/retrieve}/health"}

if [[ -n "${MAPD_GPU_DEVICES:-}" ]]; then
  [[ "$MAPD_GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "MAPD_GPU_DEVICES must be a comma-separated list of physical GPU indices" >&2
    exit 2
  }
  export CUDA_VISIBLE_DEVICES=$MAPD_GPU_DEVICES
fi
PROTOCOL_RUN_TAG=${MAPD_PROTOCOL_RUN_TAG:-r${PROTOCOL_SYNTHESIS_REVISION}}
[[ "$PROTOCOL_RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "MAPD_PROTOCOL_RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
}

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_NVCC=$(readlink -f "$(command -v nvcc)")
  export CUDA_HOME=$(dirname "$(dirname "$CUDA_NVCC")")
elif [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda
elif [[ -z "${CUDA_HOME:-}" && -x "/usr/local/cuda-${MAPD_CUDA_SERIES:-13.0}/bin/nvcc" ]]; then
  export CUDA_HOME="/usr/local/cuda-${MAPD_CUDA_SERIES:-13.0}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

retriever_is_healthy() {
  curl -fsS --connect-timeout 1 --max-time 3 "$MAPD_RETRIEVER_HEALTH_URL" >/dev/null
}

acquire_gpu_locks() {
  if [[ "${MAPD_SHARED_SERVER:-0}" == 1 && -z "${MAPD_GPU_DEVICES:-}" ]]; then
    echo "MAPD_GPU_DEVICES must be set before a GPU job on a shared server" >&2
    echo "Run 'bash mapd.sh server-preflight' after selecting currently idle physical GPU indices." >&2
    exit 1
  fi
  [[ -n "${MAPD_GPU_DEVICES:-}" ]] || return 0
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required for shared-server GPU locking" >&2
    exit 1
  }
  mkdir -p "$PROJECT_ROOT/.run"
  GPU_LOCK_FDS=()
  IFS=, read -r -a gpu_indices <<< "$MAPD_GPU_DEVICES"
  for gpu_index in "${gpu_indices[@]}"; do
    exec {gpu_lock_fd}>"$PROJECT_ROOT/.run/gpu-${gpu_index}.lock"
    if ! flock -n "$gpu_lock_fd"; then
      echo "GPU $gpu_index is already reserved by another MAPD job in this checkout" >&2
      exit 1
    fi
    GPU_LOCK_FDS+=("$gpu_lock_fd")
  done
  bash "$PROJECT_ROOT/scripts/shared_server_preflight.sh"
}

usage() {
  cat <<'EOF'
Usage: bash mapd.sh COMMAND

Commands:
  bootstrap  Rebuild or repair the complete ephemeral GPU-server environment
  server-preflight  Inspect shared-server storage, ports, CUDA, and selected GPUs
  setup      Install into the sibling veRL environment and run full verification
  verify     Run tests and the offline end-to-end smoke flow
  data-setup Download and prepare the de-duplicated 25,600-example training set
  data-sample [PERCENT] [SEED]  Create a deterministic source-stratified training subset
  data-export [source]  Validate and bundle the prepared training data; include raw Parquet with source
  wiki-setup Download wiki-18 and build/reuse the persistent SQLite FTS5 index
  wiki-start/status/logs/stop  Manage wiki-18 setup as a background job
  e5-compressed-build  Build/resume the low-memory E5 IVF-PQ index
  e5-compressed-start/status/logs/stop  Manage compressed E5 indexing
  retrieval-start/status/logs/stop  Manage the command-line retrieval service
  retrieval-smoke  Check real top-3 retrieval against normalized QA examples
  grpo-smoke Run real rollouts until rewards vary, then perform one GRPO update
  protocol-smoke  Generate a budget-capped real-teacher protocol sample
  protocol-shard OFFSET COUNT  Generate one deterministic resumable shard
  protocol-start/status/logs/stop  Manage a protocol shard background job
  protocol-export OFFSET COUNT  Create a checksummed portable shard bundle
  shallow-trace-export [ROLLOUTS] [ID]  Export one failed shallow-search trace for review
  review-traces-export OFFSET COUNT  Export two teacher negatives plus one student negative
  privileged-rollout OFFSET COUNT [MODEL]  Probe student generation with protocol PI
  privileged-start/status/logs/stop  Manage the privileged-generation probe job
  protocol-restore BUNDLE  Restore a bundle to its original artifact directory
  protocol-merge BUNDLE...  Merge and de-duplicate exported protocol bundles
  protocol-train-smoke OFFSET COUNT  Train on one real protocol shard and reload checkpoint
  protocol-train-start/status/logs/stop  Manage the real-protocol training smoke job
  protocol-checkpoint-export OFFSET COUNT  Merge the smoke checkpoint into a vLLM-loadable model
  student-eval VARIANT [COUNT] [MODEL]  Evaluate baseline/trained model on a fixed held-out cohort
  student-eval-start/status/logs/stop VARIANT  Manage a held-out evaluation job
  student-eval-compare [COUNT]  Compare paired baseline and trained evaluation runs
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
  server-preflight)
    bash "$PROJECT_ROOT/scripts/shared_server_preflight.sh"
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
  data-export)
    cd "$PROJECT_ROOT"
    DATA_EXPORT_ARGS=()
    DATA_EXPORT_SUFFIX=""
    if [[ "${2:-}" == "source" ]]; then
      DATA_EXPORT_ARGS+=(--include-source)
      DATA_EXPORT_SUFFIX="_with_source"
    elif [[ -n "${2:-}" ]]; then
      echo "usage: bash mapd.sh data-export [source]" >&2
      exit 2
    fi
    "$VERL_VENV/bin/python" scripts/data_bundle.py \
      --data-root "$PROJECT_ROOT/data" \
      --output "$PROJECT_ROOT/exports/mapd_training_data_25600_seed42${DATA_EXPORT_SUFFIX}.tar.gz" \
      --expected-revision "$MAPD_QA_REVISION" \
      "${DATA_EXPORT_ARGS[@]}"
    ;;
  data-sample)
    cd "$PROJECT_ROOT"
    PERCENT=${2:-30}
    SAMPLE_SEED=${3:-42}
    [[ "$PERCENT" =~ ^[0-9]+([.][0-9]+)?$ && "$SAMPLE_SEED" =~ ^-?[0-9]+$ ]] || {
      echo "PERCENT must be numeric and SEED must be an integer" >&2
      exit 2
    }
    PERCENT_LABEL=${PERCENT//./p}
    "$VERL_VENV/bin/python" scripts/sample_training_data.py \
      --input "$PROJECT_ROOT/data/training/mapd_train_25600.jsonl" \
      --output "$PROJECT_ROOT/data/training/mapd_train_${PERCENT_LABEL}pct_seed${SAMPLE_SEED}.jsonl" \
      --manifest "$PROJECT_ROOT/data/training/mapd_train_${PERCENT_LABEL}pct_seed${SAMPLE_SEED}.manifest.json" \
      --percent "$PERCENT" \
      --seed "$SAMPLE_SEED"
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
  e5-compressed-build)
    cd "$PROJECT_ROOT"
    E5_BUILD_ARGS=()
    [[ -z "${MAPD_E5_LIMIT:-}" ]] || E5_BUILD_ARGS+=(--limit "$MAPD_E5_LIMIT")
    "$VERL_VENV/bin/python" scripts/build_e5_compressed_index.py \
      --docstore "${MAPD_E5_DOCSTORE:-$PROJECT_ROOT/data/wiki18/index/wiki18.sqlite3}" \
      --index "${MAPD_E5_INDEX:-$PROJECT_ROOT/data/wiki18/e5_ivfpq/wiki18_e5_ivfpq.faiss}" \
      --model "${MAPD_E5_MODEL:-intfloat/e5-base-v2}" \
      --device "${MAPD_E5_DEVICE:-auto}" \
      --batch-size "${MAPD_E5_BATCH_SIZE:-128}" \
      --nlist "${MAPD_E5_NLIST:-4096}" \
      --pq-m "${MAPD_E5_PQ_M:-96}" \
      --pq-nbits "${MAPD_E5_PQ_NBITS:-8}" \
      --train-size "${MAPD_E5_TRAIN_SIZE:-262144}" \
      --nprobe "${MAPD_E5_NPROBE:-64}" \
      --checkpoint-every "${MAPD_E5_CHECKPOINT_EVERY:-1000000}" \
      --threads "${MAPD_E5_THREADS:-16}" \
      "${E5_BUILD_ARGS[@]}"
    ;;
  e5-compressed-start)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start e5-compressed \
      bash "$PROJECT_ROOT/mapd.sh" e5-compressed-build
    ;;
  e5-compressed-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status e5-compressed
    ;;
  e5-compressed-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs e5-compressed 100
    ;;
  e5-compressed-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop e5-compressed
    ;;
  retrieval-start)
    RETRIEVER_BACKEND=${MAPD_RETRIEVER_BACKEND:-sqlite-fts5}
    RETRIEVER_ARGS=(
      --backend "$RETRIEVER_BACKEND"
      --host "$MAPD_RETRIEVER_HOST"
      --port "$MAPD_RETRIEVER_PORT"
    )
    case "$RETRIEVER_BACKEND" in
      sqlite-fts5)
        RETRIEVER_ARGS+=(
          --index "${MAPD_FTS_INDEX:-$PROJECT_ROOT/data/wiki18/index/wiki18.sqlite3}"
          --query-timeout "${MAPD_RETRIEVER_QUERY_TIMEOUT:-20}"
        )
        ;;
      e5-ivfpq)
        RETRIEVER_ARGS+=(
          --index "${MAPD_E5_INDEX:-$PROJECT_ROOT/data/wiki18/e5_ivfpq/wiki18_e5_ivfpq.faiss}"
          --docstore "${MAPD_E5_DOCSTORE:-$PROJECT_ROOT/data/wiki18/index/wiki18.sqlite3}"
          --model "${MAPD_E5_MODEL:-intfloat/e5-base-v2}"
          --device "${MAPD_E5_DEVICE:-auto}"
          --nprobe "${MAPD_E5_NPROBE:-64}"
          --threads "${MAPD_E5_THREADS:-16}"
        )
        ;;
      *)
        echo "unsupported MAPD_RETRIEVER_BACKEND: $RETRIEVER_BACKEND" >&2
        exit 2
        ;;
    esac
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start retriever \
      "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/serve_retriever.py" \
      "${RETRIEVER_ARGS[@]}"
    RETRIEVER_HEALTH_WAIT=120
    [[ "$RETRIEVER_BACKEND" != "e5-ivfpq" ]] || RETRIEVER_HEALTH_WAIT=600
    RETRIEVER_PID_FILE="$PROJECT_ROOT/.run/retriever.pid"
    for ((attempt = 1; attempt <= RETRIEVER_HEALTH_WAIT; attempt++)); do
      if HEALTH_RESPONSE=$(curl -fsS --connect-timeout 1 --max-time 2 \
        "$MAPD_RETRIEVER_HEALTH_URL" 2>/dev/null); then
        printf '%s\n' "$HEALTH_RESPONSE"
        exit 0
      fi
      if [[ -f "$RETRIEVER_PID_FILE" ]]; then
        RETRIEVER_PID=$(<"$RETRIEVER_PID_FILE")
        if [[ "$RETRIEVER_PID" =~ ^[0-9]+$ ]] && ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
          echo "retriever exited before becoming healthy:" >&2
          bash "$PROJECT_ROOT/scripts/jobctl.sh" logs retriever 100 >&2 || true
          exit 1
        fi
      fi
      if (( attempt % 30 == 0 )); then
        echo "waiting for $RETRIEVER_BACKEND retriever: ${attempt}s/${RETRIEVER_HEALTH_WAIT}s"
      fi
      sleep 1
    done
    echo "retriever did not become healthy after ${RETRIEVER_HEALTH_WAIT}s:" >&2
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs retriever 100 >&2 || true
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
    TRAIN_DATA=${MAPD_TRAIN_DATA:-data/training/mapd_train_25600.jsonl}
    "$VERL_VENV/bin/python" scripts/retrieval_smoke.py \
      --url "$MAPD_RETRIEVER_URL" \
      --qa "$TRAIN_DATA" --limit "${2:-20}" --top-k 3
    ;;
  grpo-smoke)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
    "$VERL_VENV/bin/python" scripts/real_grpo_rollout.py \
      --model "$MODEL_PATH" --qa data/training/mapd_train_25600.jsonl \
      --retriever-url "$MAPD_RETRIEVER_URL"
    "$VERL_VENV/bin/python" scripts/real_grpo_optimize.py --model "$MODEL_PATH"
    ;;
  protocol-smoke)
    cd "$PROJECT_ROOT"
    TRAIN_DATA=${MAPD_TRAIN_DATA:-data/training/mapd_train_25600.jsonl}
    "$VERL_VENV/bin/python" scripts/generate_protocol.py \
      --config configs/paper_like.yaml \
      --input "$TRAIN_DATA" \
      --output "artifacts/protocol_smoke_${PROTOCOL_RUN_TAG}" \
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
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    TRAIN_DATA=${MAPD_TRAIN_DATA:-data/training/mapd_train_25600.jsonl}
    "$VERL_VENV/bin/python" scripts/generate_protocol.py \
      --config configs/paper_like.yaml \
      --input "$TRAIN_DATA" \
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
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    "$VERL_VENV/bin/python" scripts/protocol_bundle.py export \
      --source "$SHARD_DIR" \
      --output "$PROJECT_ROOT/exports/protocol_offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}.tar.gz" \
      --project-root "$PROJECT_ROOT" \
      --offset "$OFFSET" \
      --count "$COUNT"
    ;;
  shallow-trace-export)
    cd "$PROJECT_ROOT"
    TRACE_ARGS=()
    [[ -z "${2:-}" ]] || TRACE_ARGS+=(--rollouts "$2")
    [[ -z "${3:-}" ]] || TRACE_ARGS+=(--example-id "$3")
    "$VERL_VENV/bin/python" scripts/export_shallow_trace.py \
      --project-root "$PROJECT_ROOT" \
      --output-root "$PROJECT_ROOT/exports" \
      "${TRACE_ARGS[@]}"
    ;;
  review-traces-export)
    cd "$PROJECT_ROOT"
    OFFSET=${2:-0}
    COUNT=${3:-20}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    "$VERL_VENV/bin/python" scripts/export_review_traces.py \
      --project-root "$PROJECT_ROOT" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --output-root "$PROJECT_ROOT/exports"
    ;;
  privileged-rollout)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    [[ -s "$SHARD_DIR/artifacts.jsonl" ]] || {
      echo "protocol shard not found: $SHARD_DIR/artifacts.jsonl" >&2
      exit 1
    }
    if ! retriever_is_healthy; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    OUTPUT_DIR="$PROJECT_ROOT/artifacts/privileged_rollout/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    "$VERL_VENV/bin/python" scripts/privileged_branch_rollout.py \
      --model "$MODEL_PATH" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --output-dir "$OUTPUT_DIR" \
      --retriever-url "$MAPD_RETRIEVER_URL"
    ;;
  privileged-start)
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start privileged-rollout \
      bash "$PROJECT_ROOT/mapd.sh" privileged-rollout "$OFFSET" "$COUNT" "$MODEL_PATH"
    ;;
  privileged-status)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status privileged-rollout
    ;;
  privileged-logs)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs privileged-rollout 100
    ;;
  privileged-stop)
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop privileged-rollout
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
    acquire_gpu_locks
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    SHARD_DIR="$PROJECT_ROOT/artifacts/protocol_shards/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}"
    [[ -s "$SHARD_DIR/artifacts.jsonl" ]] || {
      echo "protocol shard not found: $SHARD_DIR/artifacts.jsonl" >&2
      exit 1
    }
    if ! retriever_is_healthy; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    OUTPUT_DIR="$PROJECT_ROOT/artifacts/protocol_train_smoke/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}_action_v4"
    "$VERL_VENV/bin/python" scripts/protocol_train_smoke_rollout.py \
      --model "$MODEL_PATH" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --output-dir "$OUTPUT_DIR" \
      --retriever-url "$MAPD_RETRIEVER_URL"
    "$VERL_VENV/bin/python" scripts/protocol_train_smoke_optimize.py \
      --model "$MODEL_PATH" \
      --artifacts "$SHARD_DIR/artifacts.jsonl" \
      --rollouts "$OUTPUT_DIR/rollouts.jsonl" \
      --output-dir "$OUTPUT_DIR"
    ;;
  protocol-train-start)
    OFFSET=${2:-0}
    COUNT=${3:-20}
    MODEL_PATH=${4:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
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
  protocol-checkpoint-export)
    cd "$PROJECT_ROOT"
    OFFSET=${2:-0}
    COUNT=${3:-20}
    BASE_MODEL=${4:-}
    [[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
      echo "OFFSET must be nonnegative and COUNT must be positive" >&2
      exit 2
    }
    TRAIN_OUTPUT="$PROJECT_ROOT/artifacts/protocol_train_smoke/offset_${OFFSET}_count_${COUNT}_${PROTOCOL_RUN_TAG}_action_v4"
    EXPORT_ARGS=()
    [[ -z "$BASE_MODEL" ]] || EXPORT_ARGS+=(--base-model "$BASE_MODEL")
    "$VERL_VENV/bin/python" scripts/export_smoke_checkpoint.py \
      --checkpoint "$TRAIN_OUTPUT/checkpoint-final/checkpoint.pt" \
      --output-dir "$TRAIN_OUTPUT/checkpoint-final/hf-model" \
      "${EXPORT_ARGS[@]}"
    ;;
  student-eval)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    VARIANT=${2:-baseline}
    EVAL_COUNT=${3:-200}
    MODEL_PATH=${4:-}
    EVAL_SEED=${MAPD_EVAL_SEED:-2026}
    [[ "$VARIANT" =~ ^[A-Za-z0-9._-]+$ ]] || {
      echo "VARIANT may contain only letters, digits, dot, underscore, and hyphen" >&2
      exit 2
    }
    [[ "$EVAL_COUNT" =~ ^[1-9][0-9]*$ && "$EVAL_SEED" =~ ^-?[0-9]+$ ]] || {
      echo "COUNT must be positive and MAPD_EVAL_SEED must be an integer" >&2
      exit 2
    }
    if [[ -z "$MODEL_PATH" ]]; then
      case "$VARIANT" in
        baseline)
          MODEL_PATH=${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}
          ;;
        trained)
          TRAIN_OFFSET=${MAPD_EVAL_TRAIN_OFFSET:-0}
          TRAIN_COUNT=${MAPD_EVAL_TRAIN_COUNT:-20}
          MODEL_PATH="$PROJECT_ROOT/artifacts/protocol_train_smoke/offset_${TRAIN_OFFSET}_count_${TRAIN_COUNT}_${PROTOCOL_RUN_TAG}_action_v4/checkpoint-final/hf-model"
          ;;
        *)
          echo "MODEL is required for evaluation variant '$VARIANT'" >&2
          exit 2
          ;;
      esac
    fi
    [[ -s "$MODEL_PATH/config.json" ]] || {
      echo "loadable model not found: $MODEL_PATH" >&2
      echo "for the trained variant, run: bash mapd.sh protocol-checkpoint-export 0 20" >&2
      exit 1
    }
    EVAL_DATA=${MAPD_EVAL_DATA:-$PROJECT_ROOT/data/evaluation/searchr1_test.jsonl}
    TRAIN_DATA=${MAPD_FULL_TRAIN_DATA:-$PROJECT_ROOT/data/training/mapd_train_25600.jsonl}
    [[ -s "$EVAL_DATA" ]] || { echo "held-out evaluation data not found: $EVAL_DATA" >&2; exit 1; }
    [[ -s "$TRAIN_DATA" ]] || { echo "training data not found: $TRAIN_DATA" >&2; exit 1; }
    if ! retriever_is_healthy; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    OUTPUT_DIR="$PROJECT_ROOT/artifacts/student_eval/${VARIANT}_count_${EVAL_COUNT}_seed${EVAL_SEED}_${PROTOCOL_RUN_TAG}"
    RETRIEVER_BACKEND=${MAPD_RETRIEVER_BACKEND:-sqlite-fts5}
    case "$RETRIEVER_BACKEND" in
      e5-ivfpq)
        RETRIEVER_INDEX=${MAPD_E5_INDEX:-$PROJECT_ROOT/data/wiki18/e5_ivfpq/wiki18_e5_ivfpq.faiss}
        RETRIEVER_IDENTITY_ARGS=(--retriever-nprobe "${MAPD_E5_NPROBE:-64}")
        ;;
      sqlite-fts5)
        RETRIEVER_INDEX=${MAPD_FTS_INDEX:-$PROJECT_ROOT/data/wiki18/index/wiki18.sqlite3}
        RETRIEVER_IDENTITY_ARGS=()
        ;;
      *)
        echo "unsupported MAPD_RETRIEVER_BACKEND: $RETRIEVER_BACKEND" >&2
        exit 2
        ;;
    esac
    [[ -s "$RETRIEVER_INDEX" ]] || { echo "retriever index not found: $RETRIEVER_INDEX" >&2; exit 1; }
    "$VERL_VENV/bin/python" scripts/student_eval.py \
      --model "$MODEL_PATH" \
      --qa "$EVAL_DATA" \
      --training-data "$TRAIN_DATA" \
      --output-dir "$OUTPUT_DIR" \
      --retriever-url "$MAPD_RETRIEVER_URL" \
      --retriever-backend "$RETRIEVER_BACKEND" \
      --retriever-index "$RETRIEVER_INDEX" \
      "${RETRIEVER_IDENTITY_ARGS[@]}" \
      --count "$EVAL_COUNT" \
      --seed "$EVAL_SEED"
    ;;
  student-eval-start)
    VARIANT=${2:-baseline}
    EVAL_COUNT=${3:-200}
    MODEL_PATH=${4:-}
    [[ "$VARIANT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid VARIANT" >&2; exit 2; }
    EVAL_COMMAND=(bash "$PROJECT_ROOT/mapd.sh" student-eval "$VARIANT" "$EVAL_COUNT")
    [[ -z "$MODEL_PATH" ]] || EVAL_COMMAND+=("$MODEL_PATH")
    bash "$PROJECT_ROOT/scripts/jobctl.sh" start "student-eval-$VARIANT" "${EVAL_COMMAND[@]}"
    ;;
  student-eval-status)
    VARIANT=${2:-baseline}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" status "student-eval-$VARIANT"
    ;;
  student-eval-logs)
    VARIANT=${2:-baseline}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" logs "student-eval-$VARIANT" 100
    ;;
  student-eval-stop)
    VARIANT=${2:-baseline}
    bash "$PROJECT_ROOT/scripts/jobctl.sh" stop "student-eval-$VARIANT"
    ;;
  student-eval-compare)
    cd "$PROJECT_ROOT"
    EVAL_COUNT=${2:-200}
    EVAL_SEED=${MAPD_EVAL_SEED:-2026}
    BASELINE_DIR="$PROJECT_ROOT/artifacts/student_eval/baseline_count_${EVAL_COUNT}_seed${EVAL_SEED}_${PROTOCOL_RUN_TAG}"
    TRAINED_DIR="$PROJECT_ROOT/artifacts/student_eval/trained_count_${EVAL_COUNT}_seed${EVAL_SEED}_${PROTOCOL_RUN_TAG}"
    "$VERL_VENV/bin/python" scripts/compare_student_evals.py \
      --baseline "$BASELINE_DIR/rollouts.jsonl" \
      --trained "$TRAINED_DIR/rollouts.jsonl" \
      --output "$PROJECT_ROOT/exports/student_eval_count_${EVAL_COUNT}_seed${EVAL_SEED}_${PROTOCOL_RUN_TAG}"
    ;;
  paper-examples)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
    if ! retriever_is_healthy; then
      echo "retriever is not healthy; run: bash mapd.sh retrieval-start" >&2
      exit 1
    fi
    "$VERL_VENV/bin/python" scripts/paper_examples_smoke.py \
      --model "$MODEL_PATH" \
      --retriever-url "$MAPD_RETRIEVER_URL"
    ;;
  paper-examples-start)
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
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
    acquire_gpu_locks
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/model_smoke.py" "${2:-}"
    ;;
  agent-smoke)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/agent_smoke.py" "${2:-}"
    ;;
  train-smoke)
    cd "$PROJECT_ROOT"
    acquire_gpu_locks
    MODEL_PATH=${2:-${MAPD_MODEL_PATH:-"$MAPD_MODEL_ROOT/Qwen3-1.7B"}}
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
    acquire_gpu_locks
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
