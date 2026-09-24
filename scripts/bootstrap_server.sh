#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(dirname "$PROJECT_ROOT")
VERSIONS_FILE="$PROJECT_ROOT/scripts/bootstrap_versions.env"
if [[ ! -f "$VERSIONS_FILE" ]]; then
  echo "ERROR: missing repository version file: $VERSIONS_FILE" >&2
  exit 3
fi
# shellcheck source=bootstrap_versions.env
source "$VERSIONS_FILE"

VERL_ROOT=${VERL_ROOT:-"$WORKSPACE_ROOT/verl"}
VERL_VENV=${VERL_VENV:-"$VERL_ROOT/.venv"}
MODEL_ROOT=${MAPD_MODEL_PATH:-"$HOME/models/${MAPD_MODEL_ID##*/}"}
VERL_REF=${VERL_REF:-$MAPD_VERL_REF}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$HOME/cache/uv"}
HF_HOME=${HF_HOME:-"$HOME/cache/huggingface"}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1200}
UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
SKIP_MODEL=0
SKIP_VERIFY=0
SUDO=()
if (( EUID != 0 )); then
  SUDO=(sudo)
fi

for argument in "$@"; do
  case "$argument" in
    --skip-model) SKIP_MODEL=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    *)
      echo "Unknown bootstrap option: $argument" >&2
      echo "Supported options: --skip-model --skip-verify" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$PROJECT_ROOT/logs" "$UV_CACHE_DIR" "$HF_HOME" "$(dirname "$MODEL_ROOT")"
BOOTSTRAP_LOG="$PROJECT_ROOT/logs/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

trap 'echo "BOOTSTRAP FAILED at line $LINENO. Log: $BOOTSTRAP_LOG" >&2' ERR

export VERL_ROOT VERL_VENV CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-$MAPD_CUDA_SERIES"}
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR HF_HOME HF_ENDPOINT UV_DEFAULT_INDEX UV_HTTP_TIMEOUT
export UV_CONCURRENT_DOWNLOADS PIP_INDEX_URL
export MAPD_BOOTSTRAP_SCHEMA MAPD_VERL_REF MAPD_MODEL_ID MAPD_NUMPY_VERSION
export MAPD_CUDA_SERIES MAPD_QA_REVISION MAPD_WIKI18_REVISION MAPD_WIKI18_E5_REVISION

section() {
  echo
  echo "===== $1 ====="
}

check_base_image() {
  section "0/5 base image contract"
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: /etc/os-release is unavailable; Ubuntu 24.04 is required." >&2
    exit 8
  fi

  # The checked-out MAPD repository, Bash, apt/sudo and the NVIDIA driver are
  # the only assumed inputs. Everything else is installed below.
  # shellcheck source=/etc/os-release
  source /etc/os-release
  if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
    echo "ERROR: expected Ubuntu 24.04, found ${PRETTY_NAME:-unknown}." >&2
    exit 8
  fi
  if [[ "$(uname -m)" != x86_64 ]]; then
    echo "ERROR: expected x86_64, found $(uname -m)." >&2
    exit 8
  fi
  for command_name in bash apt-get apt-cache nvidia-smi; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      echo "ERROR: base image must provide $command_name." >&2
      exit 9
    fi
  done
  if (( EUID != 0 )) && ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: a non-root account must provide sudo." >&2
    exit 9
  fi

  echo "Base image: $PRETTY_NAME ($(uname -m))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

have_system_stack() {
  [[ -f /usr/include/python3.12/Python.h ]] &&
    command -v curl >/dev/null 2>&1 &&
    command -v git >/dev/null 2>&1 &&
    command -v gcc >/dev/null 2>&1 &&
    command -v ninja >/dev/null 2>&1 &&
    [[ -x "$CUDA_HOME/bin/nvcc" ]] &&
    [[ -f "$CUDA_HOME/include/curand.h" ]]
}

install_system_stack() {
  section "1/5 system and CUDA build tools"
  if have_system_stack; then
    echo "System build stack already present."
    return
  fi

  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3.12-dev \
    ninja-build

  local cuda_suffix=${MAPD_CUDA_SERIES/./-}
  if ! apt-cache show "cuda-compiler-$cuda_suffix" >/dev/null 2>&1; then
    local keyring
    keyring=$(mktemp --suffix=.deb)
    curl -fL --retry 5 --retry-all-errors \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
      -o "$keyring"
    "${SUDO[@]}" dpkg -i "$keyring"
    rm -f -- "$keyring"
    "${SUDO[@]}" apt-get update
  fi

  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "cuda-compiler-$cuda_suffix" \
    "libcurand-dev-$cuda_suffix"

  have_system_stack || {
    echo "ERROR: system build stack is still incomplete after apt installation." >&2
    exit 11
  }
  nvcc --version | tail -n 1
  ninja --version
}

clone_verl_if_needed() {
  if [[ -d "$VERL_ROOT/.git" ]]; then
    echo "Reusing veRL checkout: $VERL_ROOT"
    return
  fi
  if [[ -e "$VERL_ROOT" ]]; then
    echo "ERROR: $VERL_ROOT exists but is not a Git checkout." >&2
    exit 20
  fi

  local attempt
  for attempt in 1 2 3; do
    echo "Cloning veRL $VERL_REF (attempt $attempt/3)..."
    if git clone --depth 1 --branch "$VERL_REF" \
      https://github.com/verl-project/verl.git "$VERL_ROOT"; then
      return
    fi
  done
  echo "ERROR: unable to clone veRL after three attempts." >&2
  exit 21
}

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    uv --version
    return
  fi
  curl -LsSf --retry 5 --retry-all-errors https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv --version
}

have_python_stack() {
  [[ -x "$VERL_VENV/bin/python" ]] &&
    "$VERL_VENV/bin/python" -c "import torch, verl, vllm" >/dev/null 2>&1
}

report_python_stack() {
  if [[ ! -e "$VERL_VENV" ]]; then
    echo "Virtual environment: MISSING ($VERL_VENV)"
    echo "A clean environment will be created from veRL's uv.lock."
  elif [[ ! -x "$VERL_VENV/bin/python" ]]; then
    echo "Virtual environment: INCOMPLETE ($VERL_VENV)"
    echo "The existing directory will be repaired from veRL's uv.lock."
  elif have_python_stack; then
    echo "Virtual environment: HEALTHY ($VERL_VENV)"
    echo "The persistent environment can be reused."
  else
    echo "Virtual environment: BROKEN ($VERL_VENV)"
    echo "Missing or unusable Python packages will be repaired from veRL's uv.lock."
  fi
}

install_python_stack() {
  section "2/5 veRL Python environment"
  clone_verl_if_needed
  install_uv_if_needed
  report_python_stack

  if have_python_stack; then
    echo "Reusing working environment: $VERL_VENV"
  else
    echo "Materializing veRL FSDP + vLLM environment..."
    (
      cd "$VERL_ROOT"
      uv sync --frozen --all-packages --extra fsdp --extra vllm \
        --no-install-package flash-attn
    )
  fi

  if ! "$VERL_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    local get_pip
    get_pip=$(mktemp --suffix=.py)
    curl -fL --retry 5 --retry-all-errors \
      https://bootstrap.pypa.io/get-pip.py -o "$get_pip"
    "$VERL_VENV/bin/python" "$get_pip"
    rm -f -- "$get_pip"
  fi

  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "$PROJECT_ROOT/requirements/server-bootstrap.txt"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --force-reinstall \
    --no-deps \
    "numpy==$MAPD_NUMPY_VERSION"

  have_python_stack || {
    echo "ERROR: torch, veRL, or vLLM is not importable." >&2
    exit 22
  }
}

install_mapd() {
  section "3/5 MAPD package"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --no-build-isolation \
    -e "$PROJECT_ROOT"
  "$VERL_VENV/bin/python" -m pip check
}

download_model() {
  section "4/5 Qwen3-1.7B model"
  if [[ "$SKIP_MODEL" == 1 ]]; then
    echo "Model download skipped by request."
    return
  fi
  if [[ -s "$MODEL_ROOT/config.json" ]]; then
    echo "Reusing model: $MODEL_ROOT"
    return
  fi
  "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/download_model.py" "$MODEL_ROOT"
}

verify_stack() {
  section "5/5 verification and manifest"
  "$VERL_VENV/bin/python" "$PROJECT_ROOT/scripts/capture_environment.py" \
    "$VERL_ROOT" "$MODEL_ROOT"
  if [[ "$SKIP_VERIFY" == 1 ]]; then
    echo "Runtime verification skipped by request."
    return
  fi
  bash "$PROJECT_ROOT/mapd.sh" verify
  if [[ "$SKIP_MODEL" == 0 ]]; then
    bash "$PROJECT_ROOT/mapd.sh" agent-smoke "$MODEL_ROOT"
  fi
}

section "MAPD reproducible server bootstrap"
echo "project: $PROJECT_ROOT"
echo "workspace: $WORKSPACE_ROOT"
echo "veRL: $VERL_ROOT"
echo "model: $MODEL_ROOT"
echo "versions: $VERSIONS_FILE"
echo "log: $BOOTSTRAP_LOG"
if [[ -s "$MODEL_ROOT/config.json" ]]; then
  echo "model cache: PRESENT"
else
  echo "model cache: MISSING (download required)"
fi

check_base_image
install_system_stack
install_python_stack
install_mapd
download_model
verify_stack

echo
echo "BOOTSTRAP OK"
echo "Activate manually only when needed: source $VERL_VENV/bin/activate"
echo "Log: $BOOTSTRAP_LOG"
