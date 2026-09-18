#!/usr/bin/env bash
# Prepare and verify an Ubuntu 22.04 training server from uv.lock.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly UV_VERSION="${UV_VERSION:-0.12.16}"
readonly MIN_FREE_GB="${MIN_FREE_GB:-10}"

RUN_TESTS=1
INSTALL_DEV=1
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
TEMP_INSTALLER=""

log() {
    printf '[training-check] %s\n' "$*"
}

warn() {
    printf '[training-check] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[training-check] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${TEMP_INSTALLER}" && -f "${TEMP_INSTALLER}" ]]; then
        rm -f -- "${TEMP_INSTALLER}"
    fi
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage: shell/check_training_server.sh [options]

Checks Ubuntu 22.04, installs uv/Python 3.11 when missing, synchronizes the
project environment exactly from uv.lock, verifies imports, and runs tests.

Options:
  --require-cuda   Fail unless PyTorch can use an NVIDIA CUDA device.
  --no-tests       Skip pytest and ruff after dependency verification.
  --runtime-only   Install runtime dependencies only; implies --no-tests.
  -h, --help       Show this help message.

Environment overrides:
  UV_VERSION=0.12.16  uv installer version used when uv is missing.
  MIN_FREE_GB=10      Minimum free disk space required before installation.
  REQUIRE_CUDA=1      Equivalent to --require-cuda.
EOF
}

while (($#)); do
    case "$1" in
        --require-cuda)
            REQUIRE_CUDA=1
            ;;
        --no-tests)
            RUN_TESTS=0
            ;;
        --runtime-only)
            INSTALL_DEV=0
            RUN_TESTS=0
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
    shift
done

[[ -f "${REPOSITORY_ROOT}/pyproject.toml" ]] || die "pyproject.toml is missing."
[[ -f "${REPOSITORY_ROOT}/uv.lock" ]] || die "uv.lock is missing."

if [[ ! -r /etc/os-release ]]; then
    die "Cannot identify the operating system: /etc/os-release is unavailable."
fi
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "This script supports Ubuntu only; detected ${ID:-unknown}."
[[ "${VERSION_ID:-}" == "22.04" ]] || die \
    "Ubuntu 22.04 is required; detected ${VERSION_ID:-unknown}."
log "Operating system: ${PRETTY_NAME:-Ubuntu 22.04}"

case "$(uname -m)" in
    x86_64 | aarch64 | arm64)
        log "Architecture: $(uname -m)"
        ;;
    *)
        die "Unsupported architecture: $(uname -m)"
        ;;
esac

free_kb="$(df -Pk "${REPOSITORY_ROOT}" | awk 'NR == 2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if [[ -z "${free_kb}" || "${free_kb}" -lt "${required_kb}" ]]; then
    die "At least ${MIN_FREE_GB} GiB of free disk space is required for PyTorch and its cache."
fi
log "Free disk space check passed (minimum ${MIN_FREE_GB} GiB)."

run_as_root() {
    if ((EUID == 0)); then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Root privileges are required to install $1, but sudo is unavailable."
    fi
}

if ! command -v curl >/dev/null 2>&1; then
    log "curl is missing; installing curl and CA certificates with apt."
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends curl ca-certificates
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    log "Found $(${UV_BIN} --version)."
else
    log "uv is missing; installing uv ${UV_VERSION} into ${HOME}/.local/bin."
    TEMP_INSTALLER="$(mktemp)"
    curl --proto '=https' --tlsv1.2 -LsSf \
        "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        -o "${TEMP_INSTALLER}"
    env UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh "${TEMP_INSTALLER}"
    export PATH="${HOME}/.local/bin:${PATH}"
    UV_BIN="$(command -v uv || true)"
    [[ -n "${UV_BIN}" ]] || die "uv installation completed but uv is not executable."
fi

cd "${REPOSITORY_ROOT}"

log "Checking that uv.lock matches pyproject.toml."
"${UV_BIN}" lock --check

if "${UV_BIN}" python find 3.11 >/dev/null 2>&1; then
    log "A compatible Python 3.11 interpreter is already available."
else
    log "Python 3.11 is missing; installing a managed Python 3.11 runtime."
    "${UV_BIN}" python install 3.11
fi

sync_arguments=(sync --frozen --python 3.11)
if ((INSTALL_DEV)); then
    sync_arguments+=(--all-groups)
else
    sync_arguments+=(--no-dev)
fi
log "Checking and installing packages exactly as pinned in uv.lock."
"${UV_BIN}" "${sync_arguments[@]}"

readonly PYTHON_BIN="${REPOSITORY_ROOT}/.venv/bin/python"
[[ -x "${PYTHON_BIN}" ]] || die "The project interpreter was not created at .venv/bin/python."

log "Verifying locked direct dependencies and their importability."
"${PYTHON_BIN}" - <<'PY'
from importlib import import_module
from importlib.metadata import version
import sys

required = {
    "gymnasium": "gymnasium",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "tensorboard": "tensorboard",
    "torch": "torch",
}

print(f"Python {sys.version.split()[0]}")
for distribution, module in required.items():
    import_module(module)
    print(f"{distribution}=={version(distribution)}")
PY

log "Running Gymnasium environment validation."
PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" - <<'PY'
from gymnasium.utils.env_checker import check_env
from env.returnEnv import ReturnEnv

check_env(ReturnEnv(), skip_render_check=True)
print("Gymnasium environment check passed.")
PY

cuda_available="$(${PYTHON_BIN} -c 'import torch; print(int(torch.cuda.is_available()))')"
if command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver detected: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
else
    warn "nvidia-smi was not found. CPU training remains available."
fi

if [[ "${cuda_available}" == "1" ]]; then
    cuda_summary="$(${PYTHON_BIN} -c 'import torch; print(f"CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}")')"
    log "PyTorch GPU check passed: ${cuda_summary}"
elif [[ "${REQUIRE_CUDA}" == "1" ]]; then
    die "CUDA was required, but torch.cuda.is_available() is false. Install/repair the NVIDIA driver and rerun."
else
    warn "PyTorch cannot use CUDA; training will run on CPU. Use --require-cuda to make this fatal."
fi

if ((RUN_TESTS)); then
    log "Running pytest."
    "${UV_BIN}" run --frozen pytest
    log "Running ruff static checks."
    "${UV_BIN}" run --frozen ruff check .
fi

log "Training server is ready."
printf '\nActivate with:\n  source %q\n' "${REPOSITORY_ROOT}/.venv/bin/activate"
printf 'Example training command:\n  python scripts/train_ppo.py --episodes 100 --max-steps 1200\n'
