#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." 
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate gfp_ranker 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -m src.select_top10 --config configs/default.yaml "$@"
echo "[DONE] Top-10 candidates: outputs/top10/top10_candidates.csv"
