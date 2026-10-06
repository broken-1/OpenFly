#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/liusongbo/OpenFly-Platform
SERVICE_DIR=/home/liusongbo/services/qwen27b-vllm
SERVICE_PYTHON=/home/liusongbo/.venvs/vllm-qwen27b/bin/python
export OPENFLY_QWEN_RUN_ROOT="${OPENFLY_QWEN_RUN_ROOT:-${PROJECT_DIR}/outputs/qwen27b_full_base/$(date +%Y%m%d_%H%M%S)}"
export OPENFLY_QWEN_MODEL_PATH=/home/liusongbo/models/Qwen3.8-27B-FP8
export OPENFLY_QWEN_BACKEND=vllm
export OPENFLY_QWEN_API_URL=http://127.0.0.1:8004/v1
export OPENFLY_QWEN_API_MODEL=qwen3.8-27b-fp8
export OPENFLY_QWEN_GPU_INDEX=1
export OPENFLY_AIRSIM_GPU_INDEX="${OPENFLY_AIRSIM_GPU_INDEX:-1}"
export OPENFLY_QWEN_WAIT_FOR_GPU=0
export OPENFLY_QWEN_CONTINUE_ON_ERROR=0
export OPENFLY_AIRSIM_EXTRA_ARGS_OVERRIDE="-RenderOffscreen -NoSound -vulkan -graphicsadapter=${OPENFLY_AIRSIM_GPU_INDEX}"
export OPENFLY_UE_EXTRA_ARGS="-RenderOffscreen -NoSound -graphicsadapter=${OPENFLY_AIRSIM_GPU_INDEX}"
mkdir -p "${OPENFLY_QWEN_RUN_ROOT}"
printf 'running\n' > "${OPENFLY_QWEN_RUN_ROOT}/run_status.txt"

finish() {
    local code=$?
    trap - EXIT
    if [[ "${code}" == 0 ]]; then
        printf 'completed\n' > "${OPENFLY_QWEN_RUN_ROOT}/run_status.txt"
    else
        printf 'failed exit_code=%s\n' "${code}" > "${OPENFLY_QWEN_RUN_ROOT}/run_status.txt"
    fi
    # Stop the service only if this run owns exactly its recorded PID.
    if [[ -n "${OPENFLY_QWEN_OWNED_SERVER_PID:-}" && -f "${SERVICE_DIR}/server.pid" ]]; then
        if [[ "$(cat "${SERVICE_DIR}/server.pid")" == "${OPENFLY_QWEN_OWNED_SERVER_PID}" ]]; then
            "${SERVICE_PYTHON}" "${SERVICE_DIR}/stop.py" || true
        fi
    fi
    exit "${code}"
}
trap finish EXIT

"${SERVICE_PYTHON}" - <<'PY'
import time
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(150):
    try:
        with opener.open('http://127.0.0.1:8004/health', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise SystemExit('27B vLLM service did not become ready')
PY
bash "${PROJECT_DIR}/run_qwen3vl_full_base.sh"
