#!/bin/bash
set -euo pipefail

# Colab/T4 quick pipeline
# Usage example:
# bash cloud_setup.sh \
#   --source "real_docs:/content/data/real_docs:-1:0" \
#   --source "face_splice:/content/data/face_splice:1:1" \
#   --source "doc_tamper:/content/data/doc_tamper:-1:1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m pip install -r requirements_cloud.txt

INDEX_CSV="$SCRIPT_DIR/normalized/index.csv"
mkdir -p "$SCRIPT_DIR/normalized" "$SCRIPT_DIR/checkpoints"

python3 prepare_index.py "$@" --out "$INDEX_CSV"

python3 train.py \
  --index-csv "$INDEX_CSV" \
  --out-dir "$SCRIPT_DIR/checkpoints" \
  --epochs 8 \
  --batch-size 32 \
  --num-workers 2 \
  --freeze-backbone

python3 evaluate_calibrate.py \
  --index-csv "$INDEX_CSV" \
  --weights "$SCRIPT_DIR/checkpoints/tamper_compact_multisignal.pth" \
  --output-meta "$SCRIPT_DIR/../inference_meta.json"

echo "\nDone. Copy files to backend/services/tamper/:"
echo "  - cloud/checkpoints/tamper_compact_multisignal.pth"
echo "  - inference_meta.json"
