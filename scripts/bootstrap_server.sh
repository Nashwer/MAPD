#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(dirname "$PROJECT_ROOT")
VERL_ROOT=${VERL_ROOT:-"$WORKSPACE_ROOT/verl"}
VERL_VENV=${VERL_VENV:-"$VERL_ROOT/.venv"}
MODEL_ROOT=${MAPD_MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}
VERL_REF=${VERL_REF:-v0.9.1}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$HOME/cache/uv"}
HF_HOME=${HF_HOME:-"$HOME/cache/huggingface"}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1200}
UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
SKIP_MODEL=0
SKIP_VERIFY=0

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

export VERL_ROOT VERL_VENV CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR HF_HOME HF_ENDPOINT UV_DEFAULT_INDEX UV_HTTP_TIMEOUT
export UV_CONCURRENT_DOWNLOADS PIP_INDEX_URL

section() {
  echo
  echo "===== $1 ====="
}

have_system_stack() {
  [[ -f /usr/include/python3.12/Python.h ]] &&
    command -v curl >/dev/null 2>&1 &&
    command -v git >/dev/null 2>&1 &&
    command -v gcc >/dev/null 2>&1 &&
    command -v ninja >/dev/null 2>&1 &&
    [[ -x /usr/local/cuda-13.0/bin/nvcc ]] &&
    [[ -f /usr/local/cuda-13.0/include/curand.h ]]
}

install_system_stack() {
  section "1/5 system and CUDA build tools"
  if have_system_stack; then
    echo "System build stack already present."
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo is required to install the system build stack." >&2
    exit 10
  fi

  if ! apt-cache show cuda-compiler-13-0 >/dev/null 2>&1; then
    local keyring
    keyring=$(mktemp --suffix=.deb)
    curl -fL --retry 5 --retry-all-errors \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
      -o "$keyring"
    sudo dpkg -i "$keyring"
    rm -f -- "$keyring"
  fi

  sudo apt-get update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3.12-dev \
    ninja-build \
    cuda-compiler-13-0 \
    libcurand-dev-13-0

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

install_python_stack() {
  section "2/5 veRL Python environment"
  clone_verl_if_needed
  install_uv_if_needed

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
    "pytest>=8,<9" \
    "pyarrow>=17,<22"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --force-reinstall \
    --no-deps \
    "numpy==2.3.5"

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
echo "log: $BOOTSTRAP_LOG"

install_system_stack
install_python_stack
install_mapd
download_model
verify_stack

echo
echo "BOOTSTRAP OK"
echo "Activate manually only when needed: source $VERL_VENV/bin/activate"
echo "Log: $BOOTSTRAP_LOG"
