# Tamper Cloud Training (Colab T4)

## 1) Install
```bash
pip install -r backend/services/tamper/cloud/requirements_cloud.txt
```

## 2) Build source-disjoint index from your own folders
```bash
python backend/services/tamper/cloud/prepare_index.py \
  --source "real_docs:/content/datasets/real_docs:-1:0" \
  --source "face_splice:/content/datasets/face_splice:1:1" \
  --source "doc_tamper:/content/datasets/doc_tamper:-1:1" \
  --out backend/services/tamper/cloud/normalized/index.csv
```

`--source` format: `name:/absolute/path:photo_label:document_label`
- `photo_label`: `1` tampered, `0` clean, `-1` unknown/not applicable
- `document_label`: `1` tampered, `0` clean

## 3) Train
```bash
python backend/services/tamper/cloud/train.py \
  --index-csv backend/services/tamper/cloud/normalized/index.csv \
  --out-dir backend/services/tamper/cloud/checkpoints \
  --epochs 8 --batch-size 32 --num-workers 2 --freeze-backbone
```

Resume:
```bash
python backend/services/tamper/cloud/train.py ... --resume backend/services/tamper/cloud/checkpoints/last.ckpt
```

## 4) Calibrate + export inference metadata
```bash
python backend/services/tamper/cloud/evaluate_calibrate.py \
  --index-csv backend/services/tamper/cloud/normalized/index.csv \
  --weights backend/services/tamper/cloud/checkpoints/tamper_compact_multisignal.pth \
  --output-meta backend/services/tamper/inference_meta.json
```

## Expected artifacts
- `backend/services/tamper/cloud/checkpoints/tamper_compact_multisignal.pth`
- `backend/services/tamper/inference_meta.json`
- `backend/services/tamper/cloud/checkpoints/training_report.json`

## Notes
- Training uses on-the-fly SBI-style self-blending plus realistic post-processing.
- Validation/test are source-disjoint by dataset source name.
- Reported metrics include balanced accuracy, AUROC, AUPRC, F1, Brier, and per-source metrics (when available).
- Final accuracy must be validated on a held-out, source-disjoint real ID-card test set.
