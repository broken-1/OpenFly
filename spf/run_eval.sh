#!/usr/bin/env bash
set -euo pipefail

spf_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${spf_project_dir}"
spf_python="${OPENFLY_PYTHON:-/home/liusongbo/miniconda3/envs/openfly/bin/python}"
export PYTHONPATH="${spf_project_dir}/.qwen_runtime:${spf_project_dir}:${spf_project_dir}/train${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${OPENFLY_QWEN_GPU_INDEX:-0}"
export OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES="${OPENFLY_AIRSIM_GPU_INDEX:-0}"
export OPENFLY_AIRSIM_EXTRA_ARGS="${OPENFLY_AIRSIM_EXTRA_ARGS:--RenderOffscreen -NoSound -opengl4 -graphicsadapter=0}"
export OPENFLY_AIRSIM_STARTUP_WAIT="${OPENFLY_AIRSIM_STARTUP_WAIT:-35}"
export OPENFLY_AIRSIM_RPC_TIMEOUT="${OPENFLY_AIRSIM_RPC_TIMEOUT:-30}"
exec "${spf_python}" -u -m spf.evaluate "$@"
