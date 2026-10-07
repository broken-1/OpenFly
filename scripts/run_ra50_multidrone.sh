#!/usr/bin/env bash
set -euo pipefail
cd /home/liusongbo/OpenFly-Platform
config="${1:?Provide an explicit committed experiment config}"
output="${2:?Provide a new output directory}"
source /home/liusongbo/miniconda3/etc/profile.d/conda.sh
conda activate openfly
export USE_TF=0 TRANSFORMERS_NO_TF=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/start_ra50_batch.py --config "$config" --output "$output"
