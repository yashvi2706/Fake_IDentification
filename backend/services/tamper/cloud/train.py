"""Train compact tamper model on Colab T4 within ~4h budget."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import TamperEvalDataset, TamperTrainDataset, collate_batch
from model import CompactTamperNet, MultiTaskCrossEntropy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--tamper-prob", type=float, default=0.5)
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_head_metrics(y_true, y_prob):
    out = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = None
        out["auprc"] = None

    out["balanced_accuracy"] = float(
        0.5
        * (
            ((y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0)
            + ((y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else 0.0)
        )
    )
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["brier"] = float(np.mean((y_prob - y_true) ** 2))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    return out


def evaluate(model, loader, device):
    model.eval()
    bucket_true = defaultdict(list)
    bucket_prob = defaultdict(list)
    by_source = defaultdict(lambda: defaultdict(lambda: {"y": [], "p": []}))

    with torch.no_grad():
        for x, labels, sources in loader:
            x = x.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            logits = model(x)
            for head, out in logits.items():
                prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                y = labels[head].cpu().numpy()
                mask = y != -1
                for i in range(len(y)):
                    if mask[i]:
                        bucket_true[head].append(int(y[i]))
                        bucket_prob[head].append(float(prob[i]))
                        by_source[head][sources[i]]["y"].append(int(y[i]))
                        by_source[head][sources[i]]["p"].append(float(prob[i]))

    metrics = {}
    for head in bucket_true:
        if len(bucket_true[head]) == 0:
            metrics[head] = {}
            continue
        metrics[head] = compute_head_metrics(bucket_true[head], bucket_prob[head])
        metrics[head]["per_source"] = {}
        for source, dd in by_source[head].items():
            if len(dd["y"]) >= 8 and len(set(dd["y"])) > 1:
                metrics[head]["per_source"][source] = compute_head_metrics(dd["y"], dd["p"])

    return metrics


def fit_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running = 0.0
    n = 0

    for x, labels, _sources in loader:
        x = x.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss, _ = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.item())
        n += 1

    return running / max(1, n)


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_ds = TamperTrainDataset(args.index_csv, split="train", image_size=args.image_size, tamper_prob=args.tamper_prob)
    val_ds = TamperEvalDataset(args.index_csv, split="val", image_size=args.image_size)
    test_ds = TamperEvalDataset(args.index_csv, split="test", image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = CompactTamperNet(pretrained=True, freeze_backbone=args.freeze_backbone).to(device)
    criterion = MultiTaskCrossEntropy()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_metric = -1.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"[INFO] resumed from {args.resume} @ epoch={start_epoch}")

    logs = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = fit_one_epoch(model, train_loader, criterion, optim, scaler, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        photo_bacc = val_metrics.get("photo_replacement", {}).get("balanced_accuracy") or 0.0
        doc_bacc = val_metrics.get("document_tamper", {}).get("balanced_accuracy") or 0.0
        avg_bacc = 0.5 * (photo_bacc + doc_bacc)

        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_metrics": val_metrics,
            "avg_balanced_accuracy": float(avg_bacc),
            "lr": float(optim.param_groups[0]["lr"]),
            "time_sec": float(time.time() - t0),
        }
        logs.append(entry)

        print(f"[E{epoch:02d}] train_loss={train_loss:.4f} val_avg_bacc={avg_bacc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.ckpt"))

        if avg_bacc > best_metric:
            best_metric = avg_bacc
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, os.path.join(args.out_dir, "best.ckpt"))
            torch.save(model.state_dict(), os.path.join(args.out_dir, "tamper_compact_multisignal.pth"))

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.ckpt"), map_location=device)["model"])
    test_metrics = evaluate(model, test_loader, device)

    report = {
        "train_args": vars(args),
        "best_val_avg_balanced_accuracy": best_metric,
        "test_metrics": test_metrics,
        "history": logs,
    }
    with open(os.path.join(args.out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[DONE] saved:")
    print(f"  - {os.path.join(args.out_dir, 'tamper_compact_multisignal.pth')}")
    print(f"  - {os.path.join(args.out_dir, 'training_report.json')}")


if __name__ == "__main__":
    main()
