#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(dirname "$PROJECT_ROOT")
VERSIONS_FILE="$PROJECT_ROOT/scripts/bootstrap_versions.env"
CU124_CONSTRAINTS="$PROJECT_ROOT/requirements/runtime-cu124.constraints.txt"
if [[ ! -f "$VERSIONS_FILE" ]]; then
  echo "ERROR: missing repository version file: $VERSIONS_FILE" >&2
  exit 3
fi
# shellcheck source=bootstrap_versions.env
source "$VERSIONS_FILE"

VERL_ROOT=${VERL_ROOT:-"$WORKSPACE_ROOT/verl"}
VERL_VENV=${VERL_VENV:-"$VERL_ROOT/.venv"}
MAPD_STORAGE_ROOT=${MAPD_STORAGE_ROOT:-"$HOME"}
MAPD_CACHE_ROOT=${MAPD_CACHE_ROOT:-"$MAPD_STORAGE_ROOT/cache"}
MODEL_ROOT=${MAPD_MODEL_PATH:-"${MAPD_MODEL_ROOT:-$MAPD_STORAGE_ROOT/models}/${MAPD_MODEL_ID##*/}"}
VERL_REF=${VERL_REF:-$MAPD_VERL_REF}
# Keep explicit VERL_REF overrides possible without accidentally enforcing the
# repository's default revision.  VERL_REVISION may be supplied alongside an
# override when an exact commit is desired.
VERL_REVISION=${VERL_REVISION:-}
if [[ -z "$VERL_REVISION" && "$VERL_REF" == "$MAPD_VERL_REF" ]]; then
  VERL_REVISION=$MAPD_VERL_REVISION
fi
UV_CACHE_DIR=${UV_CACHE_DIR:-"$MAPD_CACHE_ROOT/uv"}
HF_HOME=${HF_HOME:-"$MAPD_CACHE_ROOT/huggingface"}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$MAPD_CACHE_ROOT/pip"}
TMPDIR=${TMPDIR:-"$MAPD_STORAGE_ROOT/tmp"}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1200}
UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
SKIP_MODEL=0
SKIP_VERIFY=0
SUDO=()
if (( EUID != 0 )); then
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    SUDO=(sudo -n)
  fi
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

mkdir -p "$PROJECT_ROOT/logs" "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$(dirname "$MODEL_ROOT")"
BOOTSTRAP_LOG="$PROJECT_ROOT/logs/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

trap 'echo "BOOTSTRAP FAILED at line $LINENO. Log: $BOOTSTRAP_LOG" >&2' ERR

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_NVCC=$(readlink -f "$(command -v nvcc)")
  CUDA_HOME=$(dirname "$(dirname "$CUDA_NVCC")")
elif [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_HOME=/usr/local/cuda
else
  CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-$MAPD_CUDA_SERIES"}
fi
export MAPD_STORAGE_ROOT MAPD_CACHE_ROOT MODEL_ROOT PIP_CACHE_DIR TMPDIR
export VERL_ROOT VERL_VENV CUDA_HOME
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR HF_HOME HF_ENDPOINT UV_DEFAULT_INDEX UV_HTTP_TIMEOUT
export UV_CONCURRENT_DOWNLOADS PIP_INDEX_URL PIP_CACHE_DIR TMPDIR
export MAPD_BOOTSTRAP_SCHEMA MAPD_VERL_REF MAPD_VERL_REVISION MAPD_MODEL_ID MAPD_PYTHON_VERSION MAPD_NUMPY_VERSION
export MAPD_CUDA_SERIES MAPD_QA_REVISION MAPD_WIKI18_REVISION MAPD_WIKI18_E5_REVISION
export MAPD_RUNTIME_PROFILE MAPD_TORCH_VERSION MAPD_TORCHVISION_VERSION
export MAPD_TORCHAUDIO_VERSION MAPD_VLLM_VERSION MAPD_TRANSFORMERS_VERSION MAPD_TORCH_INDEX_URL
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

section() {
  echo
  echo "===== $1 ====="
}

check_base_image() {
  section "0/5 base image contract"
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: /etc/os-release is unavailable; Ubuntu 22.04 or 24.04 is required." >&2
    exit 8
  fi

  # Shared servers may provide the complete build stack without sudo.  Missing
  # packages are installed only when root or passwordless sudo is available.
  # shellcheck source=/etc/os-release
  source /etc/os-release
  if [[ "${ID:-}" != ubuntu || ! "${VERSION_ID:-}" =~ ^(22\.04|24\.04)$ ]]; then
    echo "ERROR: expected Ubuntu 22.04 or 24.04, found ${PRETTY_NAME:-unknown}." >&2
    exit 8
  fi
  if [[ "$(uname -m)" != x86_64 ]]; then
    echo "ERROR: expected x86_64, found $(uname -m)." >&2
    exit 8
  fi
  for command_name in bash apt-get apt-cache nvidia-smi python3; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      echo "ERROR: base image must provide $command_name." >&2
      exit 9
    fi
  done
  echo "Base image: $PRETTY_NAME ($(uname -m))"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

have_system_stack() {
  # veRL's uv environment supplies its own Python.  Requiring the system
  # Python headers here would unnecessarily reject managed shared servers
  # where all runtime dependencies are installed from binary wheels.
  command -v curl >/dev/null 2>&1 &&
    command -v git >/dev/null 2>&1 &&
    command -v gcc >/dev/null 2>&1 &&
    command -v ninja >/dev/null 2>&1 &&
    command -v cmake >/dev/null 2>&1 &&
    [[ -x "$CUDA_HOME/bin/nvcc" ]] &&
    [[ -f "$CUDA_HOME/include/curand.h" ]]
}

install_system_stack() {
  section "1/5 system and CUDA build tools"
  if have_system_stack; then
    echo "System build stack already present."
    return
  fi

  if (( EUID != 0 )) && (( ${#SUDO[@]} == 0 )); then
    echo "ERROR: the shared host is missing build prerequisites and this account has no passwordless sudo." >&2
    echo "Ask the administrator to install: build-essential ninja-build cmake CUDA compiler and cuRAND headers." >&2
    echo "Detected CUDA_HOME: $CUDA_HOME" >&2
    exit 10
  fi

  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3-dev \
    ninja-build \
    cmake

  local cuda_suffix=${MAPD_CUDA_SERIES/./-}
  if ! apt-cache show "cuda-compiler-$cuda_suffix" >/dev/null 2>&1; then
    local keyring ubuntu_repo
    ubuntu_repo="ubuntu${VERSION_ID//./}"
    keyring=$(mktemp --suffix=.deb)
    curl -fL --retry 5 --retry-all-errors \
      "https://developer.download.nvidia.com/compute/cuda/repos/$ubuntu_repo/x86_64/cuda-keyring_1.1-1_all.deb" \
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
    if [[ -n "$(git -C "$VERL_ROOT" status --porcelain --untracked-files=no)" ]]; then
      echo "ERROR: $VERL_ROOT has tracked local changes; refusing to switch it to $VERL_REF." >&2
      echo "Commit or stash those veRL changes, then rerun bootstrap." >&2
      exit 20
    fi
    local attempt current_ref fetch_succeeded=0 requested_ref
    current_ref=$(git -C "$VERL_ROOT" rev-parse HEAD)

    # A bootstrap rerun on a shared or firewalled server must not require
    # GitHub when the exact pinned checkout is already present locally.
    if [[ -n "$VERL_REVISION" && "$current_ref" == "$VERL_REVISION" ]]; then
      echo "veRL revision already matches pinned $VERL_REF ($current_ref); skipping network fetch."
      return
    fi
    if requested_ref=$(git -C "$VERL_ROOT" rev-parse --verify "$VERL_REF^{commit}" 2>/dev/null) &&
       [[ "$current_ref" == "$requested_ref" ]] &&
       { [[ -z "$VERL_REVISION" ]] || [[ "$requested_ref" == "$VERL_REVISION" ]]; }; then
      echo "veRL revision already matches local $VERL_REF ($current_ref); skipping network fetch."
      return
    fi

    echo "Local veRL revision $current_ref does not match $VERL_REF; fetching it from GitHub..."
    for attempt in 1 2 3; do
      if git -C "$VERL_ROOT" fetch --depth 1 origin "$VERL_REF"; then
        fetch_succeeded=1
        break
      fi
      echo "veRL fetch failed (attempt $attempt/3)." >&2
    done
    if (( fetch_succeeded == 0 )); then
      echo "ERROR: unable to fetch veRL $VERL_REF after three attempts." >&2
      exit 21
    fi
    requested_ref=$(git -C "$VERL_ROOT" rev-parse FETCH_HEAD)
    if [[ -n "$VERL_REVISION" && "$requested_ref" != "$VERL_REVISION" ]]; then
      echo "ERROR: veRL $VERL_REF resolved to $requested_ref, expected $VERL_REVISION." >&2
      exit 22
    fi
    if [[ "$current_ref" != "$requested_ref" ]]; then
      echo "Switching veRL from $current_ref to $VERL_REF ($requested_ref)..."
      git -C "$VERL_ROOT" checkout --detach "$requested_ref"
    else
      echo "veRL revision already matches $VERL_REF ($current_ref)."
    fi
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
      local cloned_ref
      cloned_ref=$(git -C "$VERL_ROOT" rev-parse HEAD)
      if [[ -n "$VERL_REVISION" && "$cloned_ref" != "$VERL_REVISION" ]]; then
        echo "ERROR: cloned veRL $VERL_REF at $cloned_ref, expected $VERL_REVISION." >&2
        exit 22
      fi
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
  [[ -x "$VERL_VENV/bin/python" ]] || return 1
  "$VERL_VENV/bin/python" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1 || return 1
  "$VERL_VENV/bin/python" - "$MAPD_TORCH_VERSION" "$MAPD_VLLM_VERSION" "$MAPD_CUDA_SERIES" <<'PY' >/dev/null 2>&1 || return 1
import importlib.metadata
import sys

import torch
import verl  # noqa: F401
import vllm  # noqa: F401

expected_torch, expected_vllm, expected_cuda = sys.argv[1:]
actual_torch = importlib.metadata.version("torch").split("+")[0]
actual_vllm = importlib.metadata.version("vllm")
if actual_torch != expected_torch or actual_vllm != expected_vllm:
    raise SystemExit(1)
if torch.version.cuda != expected_cuda or not torch.cuda.is_available():
    raise SystemExit(1)
PY
  "$VERL_VENV/bin/python" -m pip check >/dev/null 2>&1
}

report_python_stack() {
  if [[ ! -e "$VERL_VENV" ]]; then
    echo "Virtual environment: MISSING ($VERL_VENV)"
    echo "A clean environment will be created for $MAPD_RUNTIME_PROFILE."
  elif [[ ! -x "$VERL_VENV/bin/python" ]]; then
    echo "Virtual environment: INCOMPLETE ($VERL_VENV)"
    echo "The existing directory will be repaired for $MAPD_RUNTIME_PROFILE."
  elif have_python_stack; then
    echo "Virtual environment: HEALTHY ($VERL_VENV)"
    echo "The persistent environment can be reused."
  else
    echo "Virtual environment: BROKEN ($VERL_VENV)"
    echo "The installed package/CUDA tuple does not match $MAPD_RUNTIME_PROFILE and will be rebuilt."
  fi
}

install_cu124_stack() {
  if [[ "$MAPD_CUDA_SERIES" != 12.4 ]]; then
    echo "ERROR: $MAPD_RUNTIME_PROFILE requires MAPD_CUDA_SERIES=12.4, found $MAPD_CUDA_SERIES." >&2
    exit 22
  fi
  if [[ ! -f "$CU124_CONSTRAINTS" ]]; then
    echo "ERROR: missing runtime constraints: $CU124_CONSTRAINTS" >&2
    exit 22
  fi

  echo "Creating a clean CPython $MAPD_PYTHON_VERSION environment..."
  uv python install "$MAPD_PYTHON_VERSION"
  uv venv --clear --seed --python "$MAPD_PYTHON_VERSION" "$VERL_VENV"

  # Install the CUDA runtime tuple as one ABI-compatible unit.  vLLM 0.8.5
  # is built for torch 2.6/cu124; replacing only torch produces importable
  # packages whose compiled extensions fail at runtime.
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --index-url "$MAPD_TORCH_INDEX_URL" \
    "torch==$MAPD_TORCH_VERSION" \
    "torchvision==$MAPD_TORCHVISION_VERSION" \
    "torchaudio==$MAPD_TORCHAUDIO_VERSION"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --constraint "$CU124_CONSTRAINTS" \
    "vllm==$MAPD_VLLM_VERSION" \
    "transformers==$MAPD_TRANSFORMERS_VERSION" \
    "numpy==$MAPD_NUMPY_VERSION"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --constraint "$CU124_CONSTRAINTS" \
    --no-build-isolation \
    -e "$VERL_ROOT"
}

install_uv_lock_stack() {
  echo "Installing managed CPython $MAPD_PYTHON_VERSION..."
  uv python install "$MAPD_PYTHON_VERSION"
  echo "Materializing veRL FSDP + vLLM environment with CPython $MAPD_PYTHON_VERSION..."
  (
    cd "$VERL_ROOT"
    uv sync --python "$MAPD_PYTHON_VERSION" \
      --frozen --all-packages --extra fsdp --extra vllm \
      --no-install-package flash-attn
  )
}

install_python_stack() {
  local constraint_args=()
  if [[ "$MAPD_RUNTIME_PROFILE" == cu124-torch26-vllm085 ]]; then
    constraint_args=(--constraint "$CU124_CONSTRAINTS")
  fi
  section "2/5 veRL Python environment"
  clone_verl_if_needed
  install_uv_if_needed
  report_python_stack

  if have_python_stack; then
    echo "Reusing working environment: $VERL_VENV"
  else
    case "$MAPD_RUNTIME_PROFILE" in
      cu124-torch26-vllm085)
        install_cu124_stack
        ;;
      verl-uv-lock)
        install_uv_lock_stack
        ;;
      *)
        echo "ERROR: unsupported MAPD_RUNTIME_PROFILE=$MAPD_RUNTIME_PROFILE" >&2
        exit 22
        ;;
    esac
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
    "${constraint_args[@]}" \
    -r "$PROJECT_ROOT/requirements/server-bootstrap.txt"
  "$VERL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    "${constraint_args[@]}" \
    --force-reinstall \
    --no-deps \
    "numpy==$MAPD_NUMPY_VERSION"

  have_python_stack || {
    echo "ERROR: the pinned runtime does not pass its dependency and CUDA $MAPD_CUDA_SERIES checks." >&2
    echo "Dependency diagnostics:" >&2
    "$VERL_VENV/bin/python" -m pip check >&2 || true
    echo "CUDA diagnostics:" >&2
    "$VERL_VENV/bin/python" - <<'PY' >&2 || true
import importlib.metadata

try:
    import torch
except Exception as error:
    print(f"torch import failed: {error!r}")
else:
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

for package in ("verl", "vllm", "ray", "datasets", "protobuf"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: MISSING")
PY
    exit 22
  }

  "$VERL_VENV/bin/python" - <<'PY'
import torch
import vllm

print(f"torch: {torch.__version__}")
print(f"torch CUDA: {torch.version.cuda}")
print(f"vLLM: {vllm.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"visible GPUs: {torch.cuda.device_count()}")
PY
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
