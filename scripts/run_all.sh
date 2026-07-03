#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# GFP Ranker — one-click full pipeline
# Usage:
#   bash scripts/run_all.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.." 

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate gfp_ranker 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── GPU check ─────────────────────────────────────────────────────
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

echo "[INFO] Verifying torch CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA is not available. Run setup_env.sh first.")
PY

# ── Pipeline ──────────────────────────────────────────────────────
echo ""; echo "[STEP 1/7] Data preprocessing..."
python -m src.data --config configs/default.yaml

echo ""; echo "[STEP 2/7] Mutation processing..."
python -m src.mutation --config configs/default.yaml

echo ""; echo "[STEP 3/7] ESM2 embedding extraction..."
python -m src.embeddings_esm2 --config configs/default.yaml

echo ""; echo "[STEP 4/7] Feature building..."
python -m src.features --config configs/default.yaml

echo ""; echo "[STEP 4b/7] Data splits..."
python -m src.splits --config configs/default.yaml

echo ""; echo "[STEP 5/7] Regression baselines..."
python -m src.train_regression --config configs/default.yaml || echo "[WARN] Regression step had errors, continuing..."

echo ""; echo "[STEP 6/7] MLP ranker training..."
python -m src.train_ranker --config configs/default.yaml

echo ""; echo "[STEP 6b/7] Evaluation summary..."
python -m src.evaluate --config configs/default.yaml

echo ""; echo "[STEP 7/7] Top-10 candidate selection..."
python -m src.select_top10 --config configs/default.yaml

echo ""
echo "================================================================"
echo "  [DONE] All outputs are under outputs/"
echo "  Top-10 candidates: outputs/top10/top10_candidates.csv"
echo "  Metrics summary:   outputs/results/metrics_summary.csv"
echo "================================================================"
