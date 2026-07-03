#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# GFP Ranker — environment setup
# Creates conda env "gfp_ranker" with all required dependencies
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="gfp_ranker"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Find conda ────────────────────────────────────────────────────
if command -v mamba &>/dev/null; then
    CONDA_CMD=mamba
elif command -v conda &>/dev/null; then
    CONDA_CMD=conda
else
    echo "[ERROR] conda/mamba not found. Install miniconda first."
    exit 1
fi
echo "[INFO] Using: $CONDA_CMD"

# ── Create environment ────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Env '${ENV_NAME}' already exists, skipping creation"
else
    echo "[INFO] Creating conda env: ${ENV_NAME} (python=3.10)"
    $CONDA_CMD create -n "${ENV_NAME}" python=3.10 -y
fi

# ── Activate ──────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[INFO] Active env: $CONDA_DEFAULT_ENV"

# ── Core packages ─────────────────────────────────────────────────
echo "[INFO] Installing core packages..."
pip install -U pip setuptools wheel
pip install numpy pandas openpyxl scipy scikit-learn tqdm pyyaml joblib matplotlib seaborn
pip install biopython

# ── PyTorch with CUDA ─────────────────────────────────────────────
echo "[INFO] Installing PyTorch (CUDA 12.8 wheel, compatible with Driver 580 / CUDA 13.0)..."
# Driver 580 supports CUDA up to 13.0; PyTorch cu128 is compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Verify CUDA ───────────────────────────────────────────────────
echo "[INFO] Verifying CUDA availability..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA is not available after PyTorch install. "
                     "Check driver compatibility and re-run with a different cu version.")
print("[OK] CUDA verified.")
PY

# ── fair-esm ──────────────────────────────────────────────────────
echo "[INFO] Installing fair-esm..."
pip install fair-esm

# ── Transformers (optional, for ProtT5 ablation) ──────────────────
echo "[INFO] Installing transformers..."
pip install transformers accelerate sentencepiece protobuf || \
    echo "[WARN] transformers install failed (ProtT5 will be skipped)"

# ── LightGBM (optional baseline) ─────────────────────────────────
echo "[INFO] Installing LightGBM..."
pip install lightgbm || echo "[WARN] LightGBM install failed (LGBM baselines will be skipped)"

# ── Final check ───────────────────────────────────────────────────
echo "[INFO] Final environment check:"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA must be available"

import numpy, pandas, sklearn, scipy, yaml, tqdm
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)

try:
    import esm
    print("fair-esm: OK")
except ImportError:
    print("[WARN] fair-esm not installed")

try:
    import lightgbm
    print("lightgbm:", lightgbm.__version__)
except ImportError:
    print("[WARN] lightgbm not installed")

print("[OK] Environment ready.")
PY

echo ""
echo "================================================================"
echo "  Environment '${ENV_NAME}' is ready."
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "  Run pipeline:   bash scripts/run_all.sh"
echo "================================================================"
