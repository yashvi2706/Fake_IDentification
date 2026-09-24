#!/bin/bash
# =============================================================================
# Module 3 — Cloud Training Setup Script
# Run this ONCE on your GCP instance (T4 GPU or v5e TPU)
#
# Location: backend/services/tamper/cloud/
# =============================================================================

set -e  # Exit on any error

echo "============================================"
echo "  Module 3: Tamper Detection Cloud Setup"
echo "============================================"

# -----------------------------------------------
# Step 1: Verify Kaggle API Token
# -----------------------------------------------
echo ""
echo "[Step 1/6] Verifying Kaggle API token..."

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found!"
    echo ""
    echo "To fix this:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under the API section"
    echo "  3. Download kaggle.json"
    echo "  4. Upload it to this machine and run:"
    echo "     mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

chmod 600 ~/.kaggle/kaggle.json

# Quick validation — try listing datasets to verify the token works
echo "  Testing Kaggle API authentication..."
if kaggle datasets list --max-size 1 > /dev/null 2>&1; then
    echo "  ✓ Kaggle API token is valid"
else
    echo "  ERROR: Kaggle API token is invalid or expired!"
    echo "  Go to https://www.kaggle.com/settings and create a new token."
    exit 1
fi

# -----------------------------------------------
# Step 2: Detect Hardware
# -----------------------------------------------
echo ""
echo "[Step 2/6] Detecting hardware..."

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "  ✓ GPU detected: $GPU_NAME"
elif python3 -c "import torch_xla" 2>/dev/null; then
    echo "  ✓ TPU (v5e) detected — will use torch_xla"
else
    echo "  ⚠ No GPU/TPU detected — training will run on CPU (very slow)"
fi

# -----------------------------------------------
# Step 3: Install Dependencies
# -----------------------------------------------
echo ""
echo "[Step 3/6] Installing Python dependencies..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_cloud.txt" --quiet

echo "  ✓ Dependencies installed"

# -----------------------------------------------
# Step 4: Download Datasets
# -----------------------------------------------
echo ""
echo "[Step 4/6] Downloading datasets..."

cd "$SCRIPT_DIR"
bash download_datasets.sh

# -----------------------------------------------
# Step 5: Normalize + Precompute ELA
# -----------------------------------------------
echo ""
echo "[Step 5/6] Normalizing datasets and precomputing ELA..."

python3 normalize_datasets.py
python3 precompute_ela.py

# -----------------------------------------------
# Step 6: Train
# -----------------------------------------------
echo ""
echo "[Step 6/6] Starting training..."

python3 train.py

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/tamper_multihead_resnet18.pth"
echo ""
echo "  Next: Download the .pth file and place it in"
echo "  backend/services/tamper/tamper_multihead_resnet18.pth"
echo "============================================"
