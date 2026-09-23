#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STATE_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/jobctl.sh start NAME COMMAND [ARG ...]
  bash scripts/jobctl.sh status NAME
  bash scripts/jobctl.sh logs NAME [LINES]
  bash scripts/jobctl.sh follow NAME
  bash scripts/jobctl.sh stop NAME
  bash scripts/jobctl.sh list
EOF
}

valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

paths_for() {
  local name=$1
  PID_FILE="$STATE_DIR/$name.pid"
  LOG_FILE_POINTER="$STATE_DIR/$name.log"
  EXIT_FILE="$STATE_DIR/$name.exit"
  CMD_FILE="$STATE_DIR/$name.cmd"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(<"$PID_FILE")
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

command=${1:-}
case "$command" in
  start)
    name=${2:-}
    [[ -n "$name" ]] && valid_name "$name" || { usage; exit 2; }
    shift 2
    [[ $# -gt 0 ]] || { usage; exit 2; }
    command -v setsid >/dev/null 2>&1 || { echo "setsid is required (Ubuntu package: util-linux)" >&2; exit 1; }
    paths_for "$name"
    if is_running; then
      echo "job '$name' is already running (pid $(<"$PID_FILE"))" >&2
      exit 1
    fi
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    log_file="$LOG_DIR/${name}_${timestamp}.log"
    rm -f "$EXIT_FILE"
    printf '%q ' "$@" > "$CMD_FILE"
    printf '\n' >> "$CMD_FILE"
    printf '%s\n' "$log_file" > "$LOG_FILE_POINTER"
    cd "$PROJECT_ROOT"
    nohup setsid bash "$PROJECT_ROOT/scripts/_job_runner.sh" "$EXIT_FILE" "$@" \
      > "$log_file" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    echo "started name=$name pid=$pid log=$log_file"
    ;;
  status)
    name=${2:-}
    [[ -n "$name" ]] && valid_name "$name" || { usage; exit 2; }
    paths_for "$name"
    if is_running; then
      state=running
      exit_code=null
    else
      state=stopped
      exit_code=$(if [[ -f "$EXIT_FILE" ]]; then cat "$EXIT_FILE"; else echo null; fi)
    fi
    pid=$(if [[ -f "$PID_FILE" ]]; then cat "$PID_FILE"; else echo null; fi)
    log_file=$(if [[ -f "$LOG_FILE_POINTER" ]]; then cat "$LOG_FILE_POINTER"; else echo null; fi)
    printf '{"name":"%s","state":"%s","pid":%s,"exit_code":%s,"log":"%s"}\n' \
      "$name" "$state" "$pid" "$exit_code" "$log_file"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader
    fi
    ;;
  logs)
    name=${2:-}
    lines=${3:-100}
    [[ -n "$name" ]] && valid_name "$name" && [[ "$lines" =~ ^[0-9]+$ ]] || { usage; exit 2; }
    paths_for "$name"
    [[ -f "$LOG_FILE_POINTER" ]] || { echo "unknown job: $name" >&2; exit 1; }
    tail -n "$lines" "$(<"$LOG_FILE_POINTER")"
    ;;
  follow)
    name=${2:-}
    [[ -n "$name" ]] && valid_name "$name" || { usage; exit 2; }
    paths_for "$name"
    [[ -f "$LOG_FILE_POINTER" ]] || { echo "unknown job: $name" >&2; exit 1; }
    tail -n 100 -f "$(<"$LOG_FILE_POINTER")"
    ;;
  stop)
    name=${2:-}
    [[ -n "$name" ]] && valid_name "$name" || { usage; exit 2; }
    paths_for "$name"
    is_running || { echo "job '$name' is not running"; exit 0; }
    pid=$(<"$PID_FILE")
    process_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    case "$process_cwd" in
      "$PROJECT_ROOT"|"$PROJECT_ROOT"/*) ;;
      *) echo "refusing to stop pid $pid outside $PROJECT_ROOT" >&2; exit 1 ;;
    esac
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ "$pgid" = "$pid" ]] || { echo "refusing to stop unexpected process group $pgid" >&2; exit 1; }
    kill -TERM -- "-$pgid"
    echo "sent SIGTERM to name=$name process_group=$pgid"
    ;;
  list)
    found=0
    for pid_file in "$STATE_DIR"/*.pid; do
      [[ -e "$pid_file" ]] || continue
      found=1
      name=$(basename "$pid_file" .pid)
      bash "$0" status "$name"
    done
    [[ "$found" -eq 1 ]] || echo "no jobs"
    ;;
  *)
    usage
    exit 2
    ;;
esac
