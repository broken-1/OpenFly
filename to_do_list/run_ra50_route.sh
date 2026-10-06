#!/usr/bin/env bash
set -euo pipefail

cd /home/liusongbo/OpenFly-Platform
source /home/liusongbo/miniconda3/etc/profile.d/conda.sh
conda activate openfly

mkdir -p to_do_list/outputs
run_ts="${OPENFLY_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
result_path="to_do_list/outputs/ra50_sample0_route_${run_ts}.jsonl"
log_path="to_do_list/outputs/ra50_sample0_route_${run_ts}.log"
qwen_gpu="${OPENFLY_QWEN_GPU_INDEX:-1}"
airsim_gpu="${OPENFLY_AIRSIM_GPU_INDEX:-0}"

PYTHONPATH="/home/liusongbo/OpenFly-Platform/.qwen_runtime:/home/liusongbo/OpenFly-Platform/train:/home/liusongbo/OpenFly-Platform/to_do_list" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
USE_TF=0 \
TRANSFORMERS_NO_TF=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
RAYON_NUM_THREADS=1 \
OPENFLY_TODO_ROUTE_RESULTS="${result_path}" \
OPENFLY_TODO_ROUTE_IMAGES="to_do_list/outputs/ra50_sample0_route_${run_ts}_images" \
OPENFLY_TODO_MAX_STEP="${OPENFLY_TODO_MAX_STEP:-40}" \
OPENFLY_QWEN_MODEL_PATH=/home/liusongbo/models/Qwen3-VL-8B-Instruct \
OPENFLY_QWEN_DEVICE=cuda:0 \
OPENFLY_QWEN_ATTN_IMPLEMENTATION="${OPENFLY_QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}" \
OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES="${airsim_gpu}" \
OPENFLY_AIRSIM_EXTRA_ARGS="-RenderOffscreen -NoSound -opengl4 -graphicsadapter=0" \
OPENFLY_AIRSIM_RPC_TIMEOUT="${OPENFLY_AIRSIM_RPC_TIMEOUT:-30}" \
OPENFLY_AIRSIM_STARTUP_WAIT="${OPENFLY_AIRSIM_STARTUP_WAIT:-35}" \
CUDA_VISIBLE_DEVICES="${qwen_gpu}" \
python -u to_do_list/run_ra50_route.py 2>&1 | tee "${log_path}"
