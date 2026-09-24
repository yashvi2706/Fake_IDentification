"""Evaluate best checkpoint and export inference metadata calibration."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from dataset import TamperEvalDataset, collate_batch
from model import CompactTamperNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output-meta", default="inference_meta.json")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    return ap.parse_args()


def nll_with_temp(logits: np.ndarray, y: np.ndarray, temp: float):
    temp = max(0.1, float(temp))
    z = logits / temp
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    p = exp / np.sum(exp, axis=1, keepdims=True)
    p = np.clip(p, 1e-9, 1.0)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 1.0

    def obj(t):
        return nll_with_temp(logits, y, t)

    res = minimize_scalar(obj, bounds=(0.3, 3.0), method="bounded")
    return float(res.x if res.success else 1.0)


def collect_logits(model, loader, device):
    model.eval()
    out = {
        "photo_replacement": {"logits": [], "y": []},
        "document_tamper": {"logits": [], "y": []},
    }
    with torch.no_grad():
        for x, labels, _sources in loader:
            x = x.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            pred = model(x)
            for head in out:
                y = labels[head]
                mask = y != -1
                if mask.sum() == 0:
                    continue
                out[head]["logits"].append(pred[head][mask].cpu().numpy())
                out[head]["y"].append(y[mask].cpu().numpy())

    for head in out:
        if out[head]["logits"]:
            out[head]["logits"] = np.concatenate(out[head]["logits"], axis=0)
            out[head]["y"] = np.concatenate(out[head]["y"], axis=0)
        else:
            out[head]["logits"] = np.zeros((0, 2), dtype=np.float32)
            out[head]["y"] = np.zeros((0,), dtype=np.int64)
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TamperEvalDataset(args.index_csv, split="val", image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_batch)

    model = CompactTamperNet(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    blobs = collect_logits(model, loader, device)

    t_photo = fit_temperature(blobs["photo_replacement"]["logits"], blobs["photo_replacement"]["y"]) if len(blobs["photo_replacement"]["y"]) else 1.0
    t_doc = fit_temperature(blobs["document_tamper"]["logits"], blobs["document_tamper"]["y"]) if len(blobs["document_tamper"]["y"]) else 1.0

    meta = {
        "model_name": "compact_tamper_net_v1",
        "input_size": args.image_size,
        "ela_quality": 90,
        "ela_scale": 20,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "photo_face_aggregate": "max",
        "temperature": {"photo": round(float(t_photo), 4), "document": round(float(t_doc), 4)},
        "fusion_weights": {"photo": 0.42, "document": 0.30, "trufor": 0.24, "metadata": 0.04},
        "prob_to_percent": "round(clamp(p,0,1)*100)",
        "overall_formula": "1 - Π_i(1 - clamp(w_i*p_i,0,1))",
    }

    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] wrote {args.output_meta}")


if __name__ == "__main__":
    main()
