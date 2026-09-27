#!/usr/bin/env bash
#SBATCH --job-name=cv0-qwen35-9b
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(pwd -P)/cluster/bwunicluster3"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_command curl
require_command nvidia-smi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-9B}"
VLLM_VERSION_EXPECTED="${VLLM_VERSION:-0.25.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_ENV="${VLLM_ROOT}/.venv"
MIRAGE_ENV="${MIRAGE_REPO_ROOT}/.venv"
CV0_CONFIG="${CV0_CONFIG:-${MIRAGE_REPO_ROOT}/configs/cv0/cv0_bwunicluster_qwen3p5_9b.yaml}"

[[ -x "${VLLM_ENV}/bin/vllm" ]] || die "vLLM environment is missing; run submit_setup.sh first"
[[ -x "${MIRAGE_ENV}/bin/mirage" ]] || die "MIRAGE environment is missing; run submit_setup.sh first"
[[ -f "${CV0_CONFIG}" ]] || die "CV-0 config not found: ${CV0_CONFIG}"

export HF_HOME="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VLLM_ROOT}/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${VLLM_ROOT}/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${VLLM_ROOT}/cache/triton}"
export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export MPLBACKEND=Agg

mkdir -p "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${VLLM_ROOT}/logs" "${SCRIPT_DIR}/logs"

vllm_version="$("${VLLM_ENV}/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
[[ "${vllm_version}" == "${VLLM_VERSION_EXPECTED}" ]] \
    || die "vLLM ${vllm_version} is installed; expected ${VLLM_VERSION_EXPECTED}"
export VLLM_SERVER_VERSION="${vllm_version}"

job_id="${SLURM_JOB_ID:-manual}"
server_log="${VLLM_ROOT}/logs/vllm-${job_id}.log"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${MIRAGE_REPO_ROOT}"
printf 'Job %s on %s; starting vLLM %s\n' "${job_id}" "$(hostname)" "${vllm_version}"
nvidia-smi

"${VLLM_ENV}/bin/vllm" serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --api-key "${OPENAI_COMPATIBLE_API_KEY}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}" \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gdn-prefill-backend triton \
    --download-dir "${HF_HUB_CACHE}" \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 240); do
    if curl --silent --show-error --fail \
        -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" \
        "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited during startup. Last log lines:\n' >&2
        tail -n 100 "${server_log}" >&2 || true
        exit 1
    fi
    sleep 5
done
[[ "${ready}" == 1 ]] || die "vLLM did not become healthy within 20 minutes; see ${server_log}"

curl --silent --show-error --fail \
    -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    | tee "${VLLM_ROOT}/logs/models-${job_id}.json"
printf '\nvLLM is healthy; starting CV-0 with config: %s\n' "${CV0_CONFIG}"

set +e
"${MIRAGE_ENV}/bin/mirage" cv0 run --config "${CV0_CONFIG}"
cv0_status=$?
set -e

printf 'CV-0 process exit code: %s (0=PASS, 1=FAIL, 2=KILL)\n' "${cv0_status}"
exit "${cv0_status}"
