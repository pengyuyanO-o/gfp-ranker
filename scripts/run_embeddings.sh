#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." 
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate gfp_ranker 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] Running ESM2 embedding extraction on GPU ${CUDA_VISIBLE_DEVICES}"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA not available")
print(f"GPU: {torch.cuda.get_device_name(0)}, free: {torch.cuda.mem_get_info()[0]//1024**2}MB")
PY
python -m src.embeddings_esm2 --config configs/default.yaml "$@"
echo "[DONE] Embedding extraction complete."
