#!/usr/bin/env bash
# Evaluate a trained PPO checkpoint on one scenario, visualize it, and test multiple seeds.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

UV_BIN="${UV_BIN:-uv}"
CHECKPOINT="${CHECKPOINT:-${REPOSITORY_ROOT}/build/checkpoints/ppo.pt}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-1200}"
TEST_SEEDS="${TEST_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
readonly RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${REPOSITORY_ROOT}/build/evaluation/ppo_test-${RUN_TIMESTAMP}}"

die() {
    printf '[ppo-test] ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${UV_BIN}" == */* ]]; then
    [[ -x "${UV_BIN}" ]] || die "uv is not executable: ${UV_BIN}"
else
    UV_BIN="$(command -v "${UV_BIN}" || true)"
    [[ -n "${UV_BIN}" ]] || die "uv was not found. Install uv or set UV_BIN."
fi

[[ -f "${CHECKPOINT}" ]] || die \
    "PPO checkpoint not found: ${CHECKPOINT}. Run shell/start_ppo_training.sh first."
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer."
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be a positive integer."

read -r -a seed_values <<<"${TEST_SEEDS}"
((${#seed_values[@]} > 0)) || die "TEST_SEEDS must contain at least one seed."
for test_seed in "${seed_values[@]}"; do
    [[ "${test_seed}" =~ ^[0-9]+$ ]] || die \
        "Every TEST_SEEDS value must be a non-negative integer; received ${test_seed}."
done

# Include the fixed scenario and every batch scenario exactly once. This uses
# indexed arrays so that the script remains compatible with macOS Bash 3.2.
all_test_seeds=("${SEED}")
for test_seed in "${seed_values[@]}"; do
    already_added=0
    for existing_seed in "${all_test_seeds[@]}"; do
        if [[ "${test_seed}" == "${existing_seed}" ]]; then
            already_added=1
            break
        fi
    done
    ((already_added)) || all_test_seeds+=("${test_seed}")
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPOSITORY_ROOT}"

printf '[ppo-test] Checkpoint: %s\n' "${CHECKPOINT}"
printf '[ppo-test] Single-scenario seed: %s; max steps: %s\n' "${SEED}" "${MAX_STEPS}"
printf '[ppo-test] Test seeds: %s\n' "${all_test_seeds[*]}"
printf '[ppo-test] Output directory: %s\n' "${OUTPUT_DIR}"

readonly BATCH_RESULT="${OUTPUT_DIR}/ppo_batch_results.jsonl"
: >"${BATCH_RESULT}"

# Evaluate and visualize every deterministic test scenario selected by seed.
for test_seed in "${all_test_seeds[@]}"; do
    printf '[ppo-test] Evaluating seed %s\n' "${test_seed}" >&2
    "${UV_BIN}" run python scripts/evaluate.py ppo \
        --checkpoint "${CHECKPOINT}" \
        --seed "${test_seed}" \
        --max-steps "${MAX_STEPS}" \
        | tee "${OUTPUT_DIR}/ppo_seed${test_seed}_result.json" \
        | tee -a "${BATCH_RESULT}"

    printf '[ppo-test] Rendering animation for seed %s\n' "${test_seed}" >&2
    "${UV_BIN}" run python scripts/visualize_episode.py ppo \
        --checkpoint "${CHECKPOINT}" \
        --seed "${test_seed}" \
        --max-steps "${MAX_STEPS}" \
        --output "${OUTPUT_DIR}/ppo_seed${test_seed}.gif" \
        --snapshot-output "${OUTPUT_DIR}/ppo_seed${test_seed}.png"
done

printf '[ppo-test] Finished. Outputs: %s\n' "${OUTPUT_DIR}"
printf '[ppo-test] Batch results: %s\n' "${BATCH_RESULT}"
