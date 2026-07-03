#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." 
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate gfp_ranker 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -m src.data --config configs/default.yaml
python -m src.mutation --config configs/default.yaml
echo "[DONE] Preprocessing complete."
