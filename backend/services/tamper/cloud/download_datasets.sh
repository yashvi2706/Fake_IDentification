#!/bin/bash
# =============================================================================
# Download all 5 datasets for Module 3 training
# Run from: backend/services/tamper/cloud/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw"
mkdir -p "$RAW_DIR"

echo "============================================"
echo "  Downloading 5 datasets"
echo "  Target: $RAW_DIR"
echo "============================================"

# -----------------------------------------------
# 1. CASIA v2 (Kaggle) — ~850MB
# General image forgery: splicing + copy-move
# -----------------------------------------------
echo ""
echo "[1/5] CASIA v2 dataset..."
if [ -d "$RAW_DIR/casia" ] && [ "$(find "$RAW_DIR/casia" -type f | head -1)" ]; then
    echo "  ✓ Already downloaded, skipping"
else
    mkdir -p "$RAW_DIR/casia"
    kaggle datasets download -d divyanshgarg/casia-dataset --unzip -p "$RAW_DIR/casia"
    echo "  ✓ CASIA v2 downloaded"
fi

# -----------------------------------------------
# 2. FantasyID (Kaggle) — ~3-5GB
# ID-specific: face-swap + text-inpaint attacks
# -----------------------------------------------
echo ""
echo "[2/5] FantasyID dataset..."
if [ -d "$RAW_DIR/fantasyid" ] && [ "$(find "$RAW_DIR/fantasyid" -type f | head -1)" ]; then
    echo "  ✓ Already downloaded, skipping"
else
    mkdir -p "$RAW_DIR/fantasyid"
    if kaggle datasets download -d wricha/fantasyid --unzip -p "$RAW_DIR/fantasyid" 2>/dev/null; then
        echo "  ✓ FantasyID downloaded"
    else
        echo "  ⚠ FantasyID download failed — continuing without it"
        echo "    (Model will train on 4 remaining datasets)"
    fi
fi

# -----------------------------------------------
# 3. DocXPand-25k (Kaggle) — ~4GB
# Authentic document diversity (no forgeries)
# -----------------------------------------------
echo ""
echo "[3/5] DocXPand-25k dataset..."
if [ -d "$RAW_DIR/docxpand" ] && [ "$(find "$RAW_DIR/docxpand" -type f | head -1)" ]; then
    echo "  ✓ Already downloaded, skipping"
else
    mkdir -p "$RAW_DIR/docxpand"
    kaggle datasets download -d satishlokkoju/docxpand-25k --unzip -p "$RAW_DIR/docxpand"
    echo "  ✓ DocXPand-25k downloaded"
fi

# -----------------------------------------------
# 4. SIDTD (GitHub) — ~2GB
# Synthetic ID tampering: face-swap + text-inpaint
# -----------------------------------------------
echo ""
echo "[4/5] SIDTD dataset..."
if [ -d "$RAW_DIR/sidtd" ] && [ "$(find "$RAW_DIR/sidtd" -type f -name '*.jpg' -o -name '*.png' | head -1)" ]; then
    echo "  ✓ Already downloaded, skipping"
else
    mkdir -p "$RAW_DIR/sidtd"
    cd "$RAW_DIR/sidtd"
    
    if [ ! -d "SIDTD_Dataset" ]; then
        git clone --depth 1 https://github.com/Oriolrt/SIDTD_Dataset.git
    fi
    
    # SIDTD has a download helper — try it, but don't fail the pipeline if it breaks
    if [ -f "SIDTD_Dataset/download.py" ]; then
        python3 SIDTD_Dataset/download.py --output_dir "$RAW_DIR/sidtd/data" 2>/dev/null || true
    fi
    
    # Also try their pip-based approach if the direct script doesn't work
    if [ ! "$(find "$RAW_DIR/sidtd" -type f -name '*.jpg' -o -name '*.png' | head -1)" ]; then
        pip install sidtd --quiet 2>/dev/null || true
        python3 -c "
try:
    from sidtd import download_dataset
    download_dataset('$RAW_DIR/sidtd/data')
except Exception as e:
    print(f'SIDTD pip download failed: {e}')
    print('Will attempt to use whatever files are available.')
" 2>/dev/null || true
    fi
    
    cd "$SCRIPT_DIR"
    echo "  ✓ SIDTD processing complete"
fi

# -----------------------------------------------
# 5. Recaptured ID / BID (Kaggle) — ~1GB
# Screen/print recapture detection
# -----------------------------------------------
echo ""
echo "[5/5] Recaptured Identity Documents (BID) dataset..."
if [ -d "$RAW_DIR/recaptured_bid" ] && [ "$(find "$RAW_DIR/recaptured_bid" -type f | head -1)" ]; then
    echo "  ✓ Already downloaded, skipping"
else
    mkdir -p "$RAW_DIR/recaptured_bid"
    kaggle datasets download -d msiamh/recaptured-identity-documents --unzip -p "$RAW_DIR/recaptured_bid"
    echo "  ✓ Recaptured BID downloaded"
fi

echo ""
echo "============================================"
echo "  All downloads complete!"
echo "  Contents of $RAW_DIR:"
echo "============================================"
du -sh "$RAW_DIR"/* 2>/dev/null || true
