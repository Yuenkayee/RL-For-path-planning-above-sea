#!/usr/bin/env bash
# Configure and verify an existing Ubuntu 22.04 Python environment for training.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly MIN_FREE_GB="${MIN_FREE_GB:-10}"

RUN_TESTS=1
INSTALL_DEV=1
INSTALL_DEPENDENCIES=1
PIP_USER=0
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

usage() {
    cat <<'EOF'
Usage: shell/check_training_server.sh [options]

Uses an existing Python 3.11-3.13 environment, installs missing or incompatible
project dependencies with that interpreter's pip, verifies imports, and runs
tests. It does not install uv, install Python, or create a virtual environment.

Options:
  --python PATH    Python interpreter to configure (default: $PYTHON_BIN or python3).
  --require-cuda   Fail unless PyTorch can use an NVIDIA CUDA device.
  --no-tests       Skip pytest and ruff after dependency verification.
  --runtime-only   Install runtime dependencies only; implies --no-tests.
  --check-only     Do not install anything; only verify the existing environment.
  --user           Pass --user to pip when installing dependencies.
  -h, --help       Show this help message.

Environment overrides:
  PYTHON_BIN=python3.11  Interpreter to configure.
  MIN_FREE_GB=10        Minimum free disk space required before installation.
  REQUIRE_CUDA=1        Equivalent to --require-cuda.
EOF
}

while (($#)); do
    case "$1" in
        --python)
            (($# >= 2)) || die "--python requires an interpreter path."
            PYTHON_BIN="$2"
            shift
            ;;
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
        --check-only)
            INSTALL_DEPENDENCIES=0
            ;;
        --user)
            PIP_USER=1
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

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python interpreter is not executable: ${PYTHON_BIN}"
else
    PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
    [[ -n "${PYTHON_BIN}" ]] || die "Python interpreter was not found. Use --python PATH."
fi

"${PYTHON_BIN}" - <<'PY' || die "Python 3.11-3.13 is required. Use --python PATH to select it."
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"Unsupported Python {sys.version.split()[0]}; expected >=3.11,<3.14."
    )
print(f"[training-check] Python: {sys.version.split()[0]} ({sys.executable})")
print(f"[training-check] Environment prefix: {sys.prefix}")
PY

"${PYTHON_BIN}" -m pip --version >/dev/null 2>&1 || \
    die "pip is unavailable for ${PYTHON_BIN}; install pip for this interpreter first."

if ((INSTALL_DEPENDENCIES)); then
    free_kb="$(df -Pk "${REPOSITORY_ROOT}" | awk 'NR == 2 {print $4}')"
    required_kb="$((MIN_FREE_GB * 1024 * 1024))"
    if [[ -z "${free_kb}" || "${free_kb}" -lt "${required_kb}" ]]; then
        die "At least ${MIN_FREE_GB} GiB of free disk space is required for PyTorch and its cache."
    fi
    log "Free disk space check passed (minimum ${MIN_FREE_GB} GiB)."

    cd "${REPOSITORY_ROOT}"
    mapfile -t dependencies < <(
        INSTALL_DEV="${INSTALL_DEV}" "${PYTHON_BIN}" - <<'PY'
import os
import tomllib
from pathlib import Path

with Path("pyproject.toml").open("rb") as stream:
    project = tomllib.load(stream)

for dependency in project["project"]["dependencies"]:
    print(dependency)
if os.environ["INSTALL_DEV"] == "1":
    for dependency in project.get("dependency-groups", {}).get("dev", []):
        print(dependency)
PY
    )
    ((${#dependencies[@]})) || die "No dependencies were found in pyproject.toml."

    pip_arguments=(--disable-pip-version-check install)
    if ((PIP_USER)); then
        pip_arguments+=(--user)
    fi
    log "Installing missing or incompatible dependencies into the selected Python environment."
    "${PYTHON_BIN}" -m pip "${pip_arguments[@]}" "${dependencies[@]}"
else
    log "Check-only mode: the selected Python environment will not be modified."
fi

cd "${REPOSITORY_ROOT}"

log "Verifying declared dependency versions and imports."
INSTALL_DEV="${INSTALL_DEV}" "${PYTHON_BIN}" - <<'PY'
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys
import tomllib

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

with Path("pyproject.toml").open("rb") as stream:
    project = tomllib.load(stream)

dependencies = list(project["project"]["dependencies"])
if os.environ["INSTALL_DEV"] == "1":
    dependencies.extend(project.get("dependency-groups", {}).get("dev", []))

modules = {
    "gymnasium": "gymnasium",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pytest": "pytest",
    "ruff": "ruff",
    "tensorboard": "tensorboard",
    "torch": "torch",
}

failures = []
for declaration in dependencies:
    requirement = Requirement(declaration)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        failures.append(f"{requirement.name} is not installed")
        continue
    if requirement.specifier and installed not in requirement.specifier:
        failures.append(
            f"{requirement.name}=={installed} does not satisfy {requirement.specifier}"
        )
        continue
    module = modules.get(requirement.name.lower())
    if module is not None:
        try:
            import_module(module)
        except Exception as exc:  # report native-library and import failures clearly
            failures.append(f"cannot import {module}: {exc}")
            continue
    print(f"{requirement.name}=={installed}")

if failures:
    print("Dependency verification failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)
PY

"${PYTHON_BIN}" -m pip check

log "Running Gymnasium environment validation."
PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" - <<'PY'
from gymnasium.utils.env_checker import check_env
from env.returnEnv import ReturnEnv

check_env(ReturnEnv(), skip_render_check=True)
print("Gymnasium environment check passed.")
PY

cuda_available="$("${PYTHON_BIN}" -c 'import torch; print(int(torch.cuda.is_available()))')"
if command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver detected: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
else
    warn "nvidia-smi was not found. CPU training remains available."
fi

if [[ "${cuda_available}" == "1" ]]; then
    cuda_summary="$("${PYTHON_BIN}" -c 'import torch; print(f"CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}")')"
    log "PyTorch GPU check passed: ${cuda_summary}"
elif [[ "${REQUIRE_CUDA}" == "1" ]]; then
    die "CUDA was required, but torch.cuda.is_available() is false. Install/repair the NVIDIA driver or CUDA-enabled PyTorch and rerun."
else
    warn "PyTorch cannot use CUDA; training will run on CPU. Use --require-cuda to make this fatal."
fi

if ((RUN_TESTS)); then
    log "Running pytest."
    "${PYTHON_BIN}" -m pytest
    log "Running ruff static checks."
    "${PYTHON_BIN}" -m ruff check .
fi

log "Training server is ready. No virtual environment was created."
printf '\nUse this interpreter for training:\n  %q scripts/train_ppo.py --episodes 100 --max-steps 1200 --device auto\n' \
    "${PYTHON_BIN}"
