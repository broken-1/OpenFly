#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="/home/liusongbo/OpenFly-Platform"
CONDA_SH="/home/liusongbo/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="openfly"
QWEN_RUNTIME="${PROJECT_DIR}/.qwen_runtime"
MODEL_PATH="${OPENFLY_QWEN_MODEL_PATH:-/home/liusongbo/models/Qwen3-VL-8B-Instruct}"

RUN_TS="${OPENFLY_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OPENFLY_QWEN_RUN_ROOT:-${PROJECT_DIR}/outputs/qwen_full_base/${RUN_TS}}"
MASTER_LOG="${RUN_ROOT}/full_run.log"
SUMMARY_TSV="${RUN_ROOT}/summary.tsv"

MAX_STEP="${OPENFLY_QWEN_MAX_STEP:-100}"
HISTORY_IMAGES="${OPENFLY_QWEN_HISTORY_IMAGES:-3}"
ATTN_IMPLEMENTATION="${OPENFLY_QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"
START_FROM="${OPENFLY_QWEN_START_FROM:-}"
RESUME_SAMPLE="${OPENFLY_QWEN_START_SAMPLE:-0}"
CONTINUE_ON_ERROR="${OPENFLY_QWEN_CONTINUE_ON_ERROR:-1}"
WAIT_FOR_GPU="${OPENFLY_QWEN_WAIT_FOR_GPU:-1}"
GPU_INDEX="${OPENFLY_QWEN_GPU_INDEX:-0}"
AIRSIM_GPU_INDEX="${OPENFLY_AIRSIM_GPU_INDEX:-0}"
GPU_MIN_FREE_MIB="${OPENFLY_QWEN_GPU_MIN_FREE_MIB:-20000}"
POLL_INTERVAL_SECONDS="${OPENFLY_QWEN_POLL_INTERVAL_SECONDS:-30}"

mkdir -p "${RUN_ROOT}"
touch "${MASTER_LOG}"
if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf 'scene\tconfig\texpected_samples\tstatus\texit_code\tNE\tSR\tOSR\tSPL\tlog\tjsonl\n' > "${SUMMARY_TSV}"
fi

log() {
    local message="[$(date '+%F %T')] $*"
    printf '%s\n' "${message}" | tee -a "${MASTER_LOG}"
}

wait_for_gpu() {
    if [[ "${WAIT_FOR_GPU}" != "1" || "${OPENFLY_QWEN_BACKEND:-transformers}" == "vllm" ]]; then
        return 0
    fi

    while true; do
        local free_mib
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" | awk 'NR==1 {print $1}')"
        log "GPU ${GPU_INDEX} free memory: ${free_mib} MiB"
        if (( free_mib >= GPU_MIN_FREE_MIB )); then
            return 0
        fi
        log "Waiting ${POLL_INTERVAL_SECONDS}s for GPU ${GPU_INDEX} >= ${GPU_MIN_FREE_MIB} MiB free"
        sleep "${POLL_INTERVAL_SECONDS}"
    done
}

run_scene() {
    local scene="$1"
    local config="$2"
    local expected_samples="$3"
    local start_sample="$4"
    local append_results="$5"
    local scene_log="${RUN_ROOT}/${scene}.log"
    local scene_jsonl="${RUN_ROOT}/${scene}.jsonl"
    local status exit_code metric_line metric_scene ne sr osr spl
    local -a scene_tee_args=()

    if [[ "${append_results}" == "1" ]]; then
        scene_tee_args+=("-a")
    fi

    wait_for_gpu
    log "START scene=${scene} config=${config} expected_samples=${expected_samples} start_sample=${start_sample} append_results=${append_results}"
    log "Scene log: ${scene_log}"
    log "Step JSONL: ${scene_jsonl}"

    set +e
    env \
        PYTHONPATH="${QWEN_RUNTIME}:${PROJECT_DIR}/train" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        OPENFLY_EVAL_INFO="${config}" \
        OPENFLY_QWEN_MODEL_PATH="${MODEL_PATH}" \
        OPENFLY_QWEN_RESULTS="${scene_jsonl}" \
        OPENFLY_QWEN_MAX_SAMPLES=0 \
        OPENFLY_QWEN_START_SAMPLE="${start_sample}" \
        OPENFLY_QWEN_APPEND_RESULTS="${append_results}" \
        OPENFLY_QWEN_MAX_STEP="${MAX_STEP}" \
        OPENFLY_QWEN_HISTORY_IMAGES="${HISTORY_IMAGES}" \
        OPENFLY_QWEN_ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
        OPENFLY_AIRSIM_EXTRA_ARGS="${OPENFLY_AIRSIM_EXTRA_ARGS_OVERRIDE:--RenderOffscreen -NoSound -opengl4 -graphicsadapter=${AIRSIM_GPU_INDEX}}" \
        OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES="${AIRSIM_GPU_INDEX}" \
        USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENFLY_AIRSIM_RPC_TIMEOUT="${OPENFLY_AIRSIM_RPC_TIMEOUT:-30}" \
        OPENFLY_AIRSIM_STARTUP_WAIT=35 \
        CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
        python -u mcts/base_qwen.py 2>&1 \
        | tee "${scene_tee_args[@]}" "${scene_log}" \
        | tee -a "${MASTER_LOG}"
    exit_code=${PIPESTATUS[0]}
    set -e

    status="failed"
    if (( exit_code == 0 )) && rg -q "Total samples: ${expected_samples}$" "${scene_log}"; then
        status="complete"
    elif (( exit_code == 0 )); then
        status="incomplete"
    fi

    metric_line="$(awk -v scene="${scene}" '$1 == scene && NF >= 5 {line=$0} END {print line}' "${scene_log}")"
    metric_scene=""
    ne=""
    sr=""
    osr=""
    spl=""
    if [[ -n "${metric_line}" ]]; then
        read -r metric_scene ne sr osr spl <<< "${metric_line}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${scene}" "${config}" "${expected_samples}" "${status}" "${exit_code}" \
        "${ne}" "${sr}" "${osr}" "${spl}" "${scene_log}" "${scene_jsonl}" \
        >> "${SUMMARY_TSV}"

    log "END scene=${scene} status=${status} exit_code=${exit_code} NE=${ne:-NA} SR=${sr:-NA} OSR=${osr:-NA} SPL=${spl:-NA}"
    if [[ "${status}" != "complete" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
        return 1
    fi
    return 0
}

main() {
    cd "${PROJECT_DIR}"
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"

    if [[ ! -d "${QWEN_RUNTIME}/transformers" ]]; then
        log "Missing isolated Qwen runtime: ${QWEN_RUNTIME}"
        log "Install it before running; see run_qwen3vl_baseline.sh setup used in this project."
        return 1
    fi
    if [[ ! -d "${MODEL_PATH}" ]]; then
        log "Missing Qwen model: ${MODEL_PATH}"
        return 1
    fi

    local datasets=(
        "env_airsim_16|configs/seen_airsim_16.json|202"
        "env_airsim_18|configs/seen_airsim_18.json|203"
        "env_airsim_23|configs/seen_airsim_23.json|200"
        "env_airsim_26|configs/seen_airsim_26.json|202"
        "env_airsim_gz|configs/seen_airsim_gz.json|201"
        "env_airsim_sh|configs/seen_airsim_sh.json|202"
        "env_ue_bigcity|configs/seen_bigcity.json|182"
    )

    log "Qwen3-VL full base evaluation"
    log "run_root=${RUN_ROOT} model=${MODEL_PATH} max_step=${MAX_STEP} history_images=${HISTORY_IMAGES} attention=${ATTN_IMPLEMENTATION} qwen_gpu=${GPU_INDEX} airsim_gpu=${AIRSIM_GPU_INDEX}"
    log "start_from=${START_FROM:-beginning} resume_sample=${RESUME_SAMPLE} continue_on_error=${CONTINUE_ON_ERROR}"

    local should_run=1
    if [[ -n "${START_FROM}" ]]; then
        should_run=0
    fi

    local entry scene config expected_samples scene_start_sample scene_append_results
    for entry in "${datasets[@]}"; do
        IFS='|' read -r scene config expected_samples <<< "${entry}"
        if [[ "${scene}" == "${START_FROM}" ]]; then
            should_run=1
        fi
        if (( should_run == 0 )); then
            log "SKIP scene=${scene}; waiting for start_from=${START_FROM}"
            continue
        fi
        scene_start_sample=0
        scene_append_results=0
        if [[ "${scene}" == "${START_FROM}" && "${RESUME_SAMPLE}" != "0" ]]; then
            scene_start_sample="${RESUME_SAMPLE}"
            scene_append_results=1
        fi
        run_scene "${scene}" "${config}" "${expected_samples}" "${scene_start_sample}" "${scene_append_results}" || return $?
    done

    log "All requested scenes finished. Summary: ${SUMMARY_TSV}"
    column -t -s $'\t' "${SUMMARY_TSV}" 2>/dev/null | tee -a "${MASTER_LOG}" || true
}

set -e
main "$@"
