#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/liusongbo/OpenFly-Platform"
CONDA_SH="/home/liusongbo/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="openfly"

GPU0_MIN_FREE_MIB="${GPU0_MIN_FREE_MIB:-16384}"
GPU1_MIN_FREE_MIB="${GPU1_MIN_FREE_MIB:-16384}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-30}"

get_free_memory_mib() {
    local gpu_index="$1"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_index}" | awk 'NR==1 {print $1}'
}

wait_for_gpus() {
    while true; do
        local free0 free1 now
        free0="$(get_free_memory_mib 0)"
        free1="$(get_free_memory_mib 1)"
        now="$(date '+%F %T')"

        echo "[${now}] GPU0 free: ${free0} MiB, GPU1 free: ${free1} MiB"

        if (( free0 >= GPU0_MIN_FREE_MIB && free1 >= GPU1_MIN_FREE_MIB )); then
            echo "[${now}] Threshold reached. Starting evaluation."
            return 0
        fi

        echo "[${now}] Waiting ${POLL_INTERVAL_SECONDS}s for GPU0>=${GPU0_MIN_FREE_MIB} MiB and GPU1>=${GPU1_MIN_FREE_MIB} MiB"
        sleep "${POLL_INTERVAL_SECONDS}"
    done
}

run_eval() {
    local eval_info="$1"
    local run_name="$2"

    wait_for_gpus

    mkdir -p test outputs
    local log_file="outputs/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting ${eval_info}"
    echo "Logging to ${log_file}"

    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OPENFLY_EVAL_INFO="${eval_info}" \
    OPENFLY_MODEL_PATH=/home/liusongbo/models/openfly-agent-7b \
    CUDA_VISIBLE_DEVICES=1 \
    python train/eval.py 2>&1 | tee "${log_file}"
}

main() {
    cd "${PROJECT_DIR}"

    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"

    run_eval "configs/eval_bigcity.json" "eval_bigcity"
    run_eval "configs/seen_bigcity.json" "seen_bigcity"
}

main "$@"
