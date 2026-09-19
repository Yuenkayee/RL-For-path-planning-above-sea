#!/usr/bin/env bash
# Start PPO training with the server's existing Python environment.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EPISODES="${EPISODES:-5000}"
MAX_STEPS="${MAX_STEPS:-1200}"
NUM_ENVS="${NUM_ENVS:-8}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
CHECKPOINT="${CHECKPOINT:-${REPOSITORY_ROOT}/build/checkpoints/ppo.pt}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${REPOSITORY_ROOT}/build/logs/ppo}"

die() {
    printf '[ppo-training] ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
else
    PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
    [[ -n "${PYTHON_BIN}" ]] || die \
        "Python was not found. Activate the training environment or set PYTHON_BIN."
fi

for value_name in EPISODES MAX_STEPS NUM_ENVS PROGRESS_INTERVAL; do
    value="${!value_name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die \
        "${value_name} must be a positive integer; received ${value}."
done

mkdir -p "$(dirname -- "${CHECKPOINT}")" "${TENSORBOARD_LOG_DIR}"

readonly RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
readonly CONSOLE_LOG="${CONSOLE_LOG:-${TENSORBOARD_LOG_DIR}/training-${RUN_TIMESTAMP}.log}"
mkdir -p "$(dirname -- "${CONSOLE_LOG}")"

cd "${REPOSITORY_ROOT}"

printf '[ppo-training] Python: %s\n' "${PYTHON_BIN}"
printf '[ppo-training] Episodes: %s; max steps: %s; parallel environments: %s\n' \
    "${EPISODES}" "${MAX_STEPS}" "${NUM_ENVS}"
printf '[ppo-training] Device: %s; seed: %s\n' "${DEVICE}" "${SEED}"
printf '[ppo-training] Checkpoint: %s\n' "${CHECKPOINT}"
printf '[ppo-training] TensorBoard directory: %s\n' "${TENSORBOARD_LOG_DIR}"
printf '[ppo-training] Console log: %s\n' "${CONSOLE_LOG}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/train_ppo.py \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --num-envs "${NUM_ENVS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --checkpoint "${CHECKPOINT}" \
    --log-dir "${TENSORBOARD_LOG_DIR}" \
    "$@" 2>&1 | tee "${CONSOLE_LOG}"
