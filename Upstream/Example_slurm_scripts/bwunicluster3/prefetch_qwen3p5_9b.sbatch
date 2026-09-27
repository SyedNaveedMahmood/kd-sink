#!/usr/bin/env bash
#SBATCH --job-name=qwen35-fetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(pwd -P)/cluster/bwunicluster3"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
VLLM_ENV="${VLLM_ROOT}/.venv"
[[ -x "${VLLM_ENV}/bin/hf" ]] || die "vLLM environment is missing; run submit_setup.sh first"

export HF_HOME="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

snapshot_path="$("${VLLM_ENV}/bin/hf" download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" --cache-dir "${HF_HUB_CACHE}")"
printf '%s\n' "${snapshot_path}" > "${VLLM_ROOT}/model_snapshot_path.txt"
printf 'Model ready: %s at %s\n' "${MODEL_ID}@${MODEL_REVISION}" "${snapshot_path}"
