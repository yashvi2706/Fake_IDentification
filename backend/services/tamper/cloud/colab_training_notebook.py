"""
Colab T4 training recipe (copy blocks into notebook cells).

1) Clone project and install deps
--------------------------------
!pip install -q torch torchvision scikit-learn scipy opencv-python-headless

2) Prepare dataset folders
--------------------------
# Put your image folders in Drive or /content and map labels:
# --source "name:/abs/path:photo_label:document_label"
# photo_label: 1 tampered, 0 clean, -1 unknown
# document_label: 1 tampered, 0 clean

3) Build source-disjoint index
------------------------------
!python backend/services/tamper/cloud/prepare_index.py \
  --source "real_docs:/content/datasets/real_docs:-1:0" \
  --source "face_splice:/content/datasets/face_splice:1:1" \
  --source "doc_tamper:/content/datasets/doc_tamper:-1:1" \
  --out backend/services/tamper/cloud/normalized/index.csv

4) Train (mixed precision, resume supported)
--------------------------------------------
!python backend/services/tamper/cloud/train.py \
  --index-csv backend/services/tamper/cloud/normalized/index.csv \
  --out-dir backend/services/tamper/cloud/checkpoints \
  --epochs 8 --batch-size 32 --num-workers 2 --freeze-backbone

# Resume example:
# !python ... --resume backend/services/tamper/cloud/checkpoints/last.ckpt

5) Calibrate and export inference metadata
-----------------------------------------
!python backend/services/tamper/cloud/evaluate_calibrate.py \
  --index-csv backend/services/tamper/cloud/normalized/index.csv \
  --weights backend/services/tamper/cloud/checkpoints/tamper_compact_multisignal.pth \
  --output-meta backend/services/tamper/inference_meta.json

6) Copy artifacts into service folder
------------------------------------
# Expected checkpoint filename:
# backend/services/tamper/cloud/checkpoints/tamper_compact_multisignal.pth
"""
