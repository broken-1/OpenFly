#!/usr/bin/env bash
set -euo pipefail

cd /home/liusongbo/OpenFly-Platform
source /home/liusongbo/miniconda3/etc/profile.d/conda.sh
conda activate openfly

mkdir -p outputs/mcts_eval
run_ts="${OPENFLY_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
log_path="outputs/mcts_eval/qwen3vl8b_mcts_random50_${run_ts}.log"
jsonl_path="outputs/mcts_eval/qwen3vl8b_mcts_random50_${run_ts}.jsonl"
qwen_gpu="${OPENFLY_QWEN_GPU_INDEX:-1}"
airsim_gpu="${OPENFLY_AIRSIM_GPU_INDEX:-1}"

PYTHONPATH="/home/liusongbo/OpenFly-Platform/.qwen_runtime:/home/liusongbo/OpenFly-Platform/train" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
OPENFLY_EVAL_INFO=configs/seen_airsim_23_random50_seed20260609.json \
OPENFLY_QWEN_MODEL_PATH=/home/liusongbo/models/Qwen3-VL-8B-Instruct \
OPENFLY_MCTS_RESULTS="${jsonl_path}" \
OPENFLY_MCTS_TOP_K="${OPENFLY_MCTS_TOP_K:-3}" \
OPENFLY_MCTS_SEARCH_DEPTH="${OPENFLY_MCTS_SEARCH_DEPTH:-2}" \
OPENFLY_MCTS_NUM_SIMULATIONS="${OPENFLY_MCTS_NUM_SIMULATIONS:-12}" \
OPENFLY_MCTS_MAX_SAMPLES="${OPENFLY_MCTS_MAX_SAMPLES:-50}" \
OPENFLY_MCTS_MAX_STEP="${OPENFLY_MCTS_MAX_STEP:-40}" \
OPENFLY_QWEN_ATTN_IMPLEMENTATION="${OPENFLY_QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}" \
OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES="${airsim_gpu}" \
OPENFLY_AIRSIM_EXTRA_ARGS="-RenderOffscreen -NoSound -opengl4 -graphicsadapter=0" \
OPENFLY_AIRSIM_RPC_TIMEOUT="${OPENFLY_AIRSIM_RPC_TIMEOUT:-30}" \
OPENFLY_AIRSIM_STARTUP_WAIT="${OPENFLY_AIRSIM_STARTUP_WAIT:-35}" \
CUDA_VISIBLE_DEVICES="${qwen_gpu}" \
python -u mcts/eval_mcts.py 2>&1 | tee "${log_path}"
