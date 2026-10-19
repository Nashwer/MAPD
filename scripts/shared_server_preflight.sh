#!/usr/bin/env bash
set -euo pipefail

# Read-only deployment guard for shared GPU hosts.  It never reserves a GPU;
# it only refuses an explicitly selected device that is already busy.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAPD_STORAGE_ROOT=${MAPD_STORAGE_ROOT:-"$HOME"}
MAPD_CACHE_ROOT=${MAPD_CACHE_ROOT:-"$MAPD_STORAGE_ROOT/cache"}
MAPD_MODEL_ROOT=${MAPD_MODEL_ROOT:-"$MAPD_STORAGE_ROOT/models"}
MAPD_RETRIEVER_HOST=${MAPD_RETRIEVER_HOST:-127.0.0.1}
MAPD_RETRIEVER_PORT=${MAPD_RETRIEVER_PORT:-8000}
MAPD_RETRIEVER_HEALTH_URL=${MAPD_RETRIEVER_HEALTH_URL:-"http://$MAPD_RETRIEVER_HOST:$MAPD_RETRIEVER_PORT/health"}
MAPD_GPU_MAX_USED_MIB=${MAPD_GPU_MAX_USED_MIB:-1024}
MAPD_GPU_MAX_UTILIZATION=${MAPD_GPU_MAX_UTILIZATION:-20}
MAPD_MIN_FREE_GIB=${MAPD_MIN_FREE_GIB:-200}

failures=0
warn() { printf 'WARN: %s\n' "$*" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; failures=$((failures + 1)); }
section() { printf '\n===== %s =====\n' "$1"; }

section "host"
printf 'user=%s host=%s project=%s\n' "$(id -un)" "$(hostname)" "$PROJECT_ROOT"
if [[ -r /etc/os-release ]]; then
  # shellcheck source=/etc/os-release
  source /etc/os-release
  printf 'os=%s architecture=%s\n' "${PRETTY_NAME:-unknown}" "$(uname -m)"
  [[ "${ID:-}" == ubuntu ]] || warn "the maintained deployment path targets Ubuntu"
  case "${VERSION_ID:-}" in
    22.04|24.04) ;;
    *) warn "Ubuntu 22.04/24.04 are tested; found ${VERSION_ID:-unknown}" ;;
  esac
fi

section "storage and memory"
storage_probe=$MAPD_STORAGE_ROOT
for path in "$MAPD_STORAGE_ROOT" "$MAPD_CACHE_ROOT" "$MAPD_MODEL_ROOT"; do
  probe=$path
  while [[ ! -e "$probe" && "$probe" != / ]]; do probe=$(dirname "$probe"); done
  if [[ ! -d "$probe" ]]; then
    fail "cannot resolve an existing parent for $path"
    continue
  fi
  [[ "$path" != "$MAPD_STORAGE_ROOT" ]] || storage_probe=$probe
  [[ -w "$probe" ]] || fail "$probe is not writable for $(id -un)"
  printf 'target=%s existing_parent=%s\n' "$path" "$probe"
  df -hP "$probe" | tail -n 1
done
free -h || true
available_kib=$(df -Pk "$storage_probe" 2>/dev/null | awk 'NR==2 {print $4}')
if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
  available_gib=$((available_kib / 1024 / 1024))
  (( available_gib >= MAPD_MIN_FREE_GIB )) ||
    fail "only ${available_gib} GiB free under $MAPD_STORAGE_ROOT; require ${MAPD_MIN_FREE_GIB} GiB"
fi

section "runtime"
for command_name in bash git curl python3 gcc g++ ninja cmake nvidia-smi; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%-12s %s\n' "$command_name" "$(command -v "$command_name")"
  else
    warn "$command_name is unavailable"
  fi
done
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
else
  warn "nvcc is unavailable; inference may work but CUDA extension builds may fail"
fi
printf 'python=%s\n' "$(python3 --version 2>&1 || true)"
printf 'open_files_limit=%s\n' "$(ulimit -n)"

section "retriever endpoint"
if command -v ss >/dev/null 2>&1 &&
  ss -H -ltn | awk '{print $4}' | grep -Eq "(^|[:.])${MAPD_RETRIEVER_PORT}$"; then
  if curl -fsS --connect-timeout 1 --max-time 2 "$MAPD_RETRIEVER_HEALTH_URL" >/dev/null 2>&1; then
    printf 'port=%s state=mapd-retriever-already-healthy url=%s\n' \
      "$MAPD_RETRIEVER_PORT" "$MAPD_RETRIEVER_HEALTH_URL"
  else
    fail "port $MAPD_RETRIEVER_PORT is occupied by another service"
  fi
else
  printf 'port=%s state=available host=%s\n' "$MAPD_RETRIEVER_PORT" "$MAPD_RETRIEVER_HOST"
fi

section "gpu inventory"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi is required for MAPD GPU training"
else
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits
fi

if [[ -z "${MAPD_GPU_DEVICES:-}" ]]; then
  warn "MAPD_GPU_DEVICES is unset; set explicit physical indices before a shared-host GPU job"
else
  [[ "$MAPD_GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail \
    "MAPD_GPU_DEVICES must be a comma-separated list of physical indices"
  section "selected gpu guard"
  IFS=, read -r -a selected_devices <<< "$MAPD_GPU_DEVICES"
  for gpu_index in "${selected_devices[@]}"; do
    row=$(nvidia-smi -i "$gpu_index" \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -z "$row" ]]; then
      fail "GPU $gpu_index does not exist or is not visible"
      continue
    fi
    IFS=, read -r index name used total utilization <<< "$row"
    index=${index//[[:space:]]/}
    used=${used//[[:space:]]/}
    total=${total//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    printf 'gpu=%s name=%s used_mib=%s total_mib=%s utilization=%s%%\n' \
      "$index" "${name# }" "$used" "$total" "$utilization"
    if (( used > MAPD_GPU_MAX_USED_MIB || utilization > MAPD_GPU_MAX_UTILIZATION )); then
      fail "GPU $gpu_index exceeds the shared-host idle limits (${MAPD_GPU_MAX_USED_MIB} MiB, ${MAPD_GPU_MAX_UTILIZATION}%)"
    fi
    while IFS=, read -r pid process_used; do
      pid=${pid//[[:space:]]/}
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
      printf '  process pid=%s owner=%s used_mib=%s\n' "$pid" "${owner:-unknown}" "${process_used//[[:space:]]/}"
    done < <(nvidia-smi -i "$gpu_index" --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
  done
fi

section "result"
if (( failures > 0 )); then
  printf 'SERVER PREFLIGHT FAILED (%d issue(s))\n' "$failures" >&2
  exit 1
fi
printf 'SERVER PREFLIGHT OK\n'
