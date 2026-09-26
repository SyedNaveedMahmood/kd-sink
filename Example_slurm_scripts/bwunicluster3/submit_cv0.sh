#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
require_command sbatch

[[ -x "${VLLM_ROOT}/.venv/bin/vllm" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"
[[ -x "${MIRAGE_REPO_ROOT}/.venv/bin/mirage" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"
mkdir -p "${SCRIPT_DIR}/logs" "${VLLM_ROOT}/logs"

common_args=(--chdir="${MIRAGE_REPO_ROOT}")
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    common_args+=(--account="${SBATCH_ACCOUNT}")
fi

dependency_args=()
if [[ "${SKIP_MODEL_PREFETCH:-0}" != 1 ]]; then
    prefetch_id="$(sbatch --parsable "${common_args[@]}" \
        --partition="${SBATCH_CPU_PARTITION:-cpu}" \
        "${SCRIPT_DIR}/prefetch_qwen3p5_9b.sbatch")"
    prefetch_id="${prefetch_id%%;*}"
    dependency_args=(--dependency="afterok:${prefetch_id}")
    printf 'Submitted model prefetch job: %s\n' "${prefetch_id}"
fi

gpu_id="$(sbatch --parsable "${common_args[@]}" "${dependency_args[@]}" \
    --partition="${SBATCH_GPU_PARTITION:-gpu_h100}" \
    --time="${SBATCH_GPU_TIME:-48:00:00}" \
    "${SCRIPT_DIR}/cv0_qwen3p5_9b.sbatch")"
gpu_id="${gpu_id%%;*}"
printf 'Submitted CV-0 GPU job: %s\n' "${gpu_id}"
printf 'Monitor with: squeue -j %s\n' "${gpu_id}"
