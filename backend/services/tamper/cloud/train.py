"""
train.py — Multi-head ResNet18 training script for tamper detection.

Location: backend/services/tamper/cloud/

Trains on pre-cached ELA images with masked per-head loss.
Designed for Google Cloud T4 GPU or v5e TPU.

Usage:
    python train.py
    python train.py --epochs 12 --batch-size 64 --lr 0.0005
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import MultiHeadResNet18, masked_cross_entropy_loss, compute_per_head_accuracy
from dataset import MultiHeadTamperDataset, collate_multihead, TRAIN_TRANSFORM, VAL_TRANSFORM

SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-head tamper detection model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output dir")
    parser.add_argument("--labels-csv", type=str, default=None, help="Path to labels.csv")
    parser.add_argument("--ela-cache-dir", type=str, default=None, help="Path to ELA cache dir")
    return parser.parse_args()


def detect_device():
    """Detect best available device: CUDA GPU, XLA TPU, or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device: {gpu_name} (CUDA)")
        return device

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("  Device: TPU v5e (XLA)")
        return device
    except ImportError:
        pass

    print("  Device: CPU (training will be slow!)")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch. Returns average loss and per-head accuracies."""
    model.train()
    total_loss = 0.0
    all_head_correct = {"photo_replacement": 0, "text_manipulation": 0, "compression_anomaly": 0}
    all_head_total = {"photo_replacement": 0, "text_manipulation": 0, "compression_anomaly": 0}
    num_batches = 0

    for batch_tensors, batch_labels in dataloader:
        batch_tensors = batch_tensors.to(device)
        labels_device = {k: v.to(device) for k, v in batch_labels.items()}

        optimizer.zero_grad()
        logits = model(batch_tensors)
        loss, per_head_loss = masked_cross_entropy_loss(logits, labels_device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Track per-head accuracy
        accs = compute_per_head_accuracy(logits, labels_device)
        for head_name in accs:
            mask = labels_device[head_name] != -1
            n = mask.sum().item()
            if accs[head_name] is not None and n > 0:
                all_head_correct[head_name] += int(accs[head_name] * n)
                all_head_total[head_name] += n

    avg_loss = total_loss / max(num_batches, 1)
    head_accs = {}
    for head_name in all_head_correct:
        if all_head_total[head_name] > 0:
            head_accs[head_name] = all_head_correct[head_name] / all_head_total[head_name]
        else:
            head_accs[head_name] = None

    return avg_loss, head_accs


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model. Returns average loss and per-head accuracies."""
    model.eval()
    total_loss = 0.0
    all_head_correct = {"photo_replacement": 0, "text_manipulation": 0, "compression_anomaly": 0}
    all_head_total = {"photo_replacement": 0, "text_manipulation": 0, "compression_anomaly": 0}
    num_batches = 0

    for batch_tensors, batch_labels in dataloader:
        batch_tensors = batch_tensors.to(device)
        labels_device = {k: v.to(device) for k, v in batch_labels.items()}

        logits = model(batch_tensors)
        loss, _ = masked_cross_entropy_loss(logits, labels_device)

        total_loss += loss.item()
        num_batches += 1

        accs = compute_per_head_accuracy(logits, labels_device)
        for head_name in accs:
            mask = labels_device[head_name] != -1
            n = mask.sum().item()
            if accs[head_name] is not None and n > 0:
                all_head_correct[head_name] += int(accs[head_name] * n)
                all_head_total[head_name] += n

    avg_loss = total_loss / max(num_batches, 1)
    head_accs = {}
    for head_name in all_head_correct:
        if all_head_total[head_name] > 0:
            head_accs[head_name] = all_head_correct[head_name] / all_head_total[head_name]
        else:
            head_accs[head_name] = None

    return avg_loss, head_accs


def main():
    args = parse_args()

    print("============================================")
    print("  Multi-Head Tamper Detection Training")
    print("============================================")
    print()

    # Paths — all relative to this script's directory
    labels_csv = args.labels_csv or str(SCRIPT_DIR / "normalized" / "labels.csv")
    ela_cache_dir = args.ela_cache_dir or str(SCRIPT_DIR / "normalized" / "ela_cache")
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (SCRIPT_DIR / "checkpoints")

    if not os.path.exists(labels_csv):
        print(f"ERROR: {labels_csv} not found!")
        print("Run normalize_datasets.py first.")
        sys.exit(1)

    if not os.path.exists(ela_cache_dir):
        print(f"ERROR: {ela_cache_dir} not found!")
        print("Run precompute_ela.py first.")
        sys.exit(1)

    # Device
    device = detect_device()

    # Load full dataset with train transforms (we'll split later)
    print()
    print("Loading dataset...")
    full_dataset = MultiHeadTamperDataset(labels_csv, ela_cache_dir, transform=TRAIN_TRANSFORM)

    if len(full_dataset) == 0:
        print("ERROR: No samples found! Check normalize and precompute steps.")
        sys.exit(1)

    # Print distribution
    print()
    print("  Source distribution:")
    for src, count in sorted(full_dataset.get_source_distribution().items()):
        print(f"    {src}: {count}")

    print()
    print("  Label distribution:")
    for head, counts in full_dataset.get_label_distribution().items():
        print(f"    {head}: pos={counts[1]}, neg={counts[0]}, masked={counts[-1]}")

    # Train/Val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    print(f"\n  Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_multihead,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_multihead,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    print("\nInitializing MultiHeadResNet18 (ImageNet pretrained)...")
    model = MultiHeadResNet18(pretrained=True).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    # Checkpoint dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs | batch_size={args.batch_size} | lr={args.lr}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    training_log = []

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_accs = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_accs = validate(model, val_loader, device)

        # LR scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": {k: round(v, 4) if v is not None else None for k, v in train_accs.items()},
            "val_acc": {k: round(v, 4) if v is not None else None for k, v in val_accs.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(epoch_time, 1),
        }
        training_log.append(log_entry)

        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        for head_name in ["photo_replacement", "text_manipulation", "compression_anomaly"]:
            ta = train_accs.get(head_name)
            va = val_accs.get(head_name)
            ta_str = f"{ta:.3f}" if ta is not None else "N/A"
            va_str = f"{va:.3f}" if va is not None else "N/A"
            print(f"  {head_name:25s} | Train Acc: {ta_str} | Val Acc: {va_str}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "tamper_multihead_resnet18.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ Best model saved (val_loss={val_loss:.4f})")

        # Also save periodic checkpoints
        if (epoch + 1) % 3 == 0 or (epoch + 1) == args.epochs:
            periodic_path = ckpt_dir / f"tamper_multihead_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_path)

        print()

    # Save training log
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print("=" * 60)
    print("  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {ckpt_dir / 'tamper_multihead_resnet18.pth'}")
    print(f"  Training log: {log_path}")
    print()
    print("  Next steps:")
    print("  1. Download tamper_multihead_resnet18.pth")
    print("  2. Place in backend/services/tamper/")
    print("  3. The inference service will auto-detect it")
    print("=" * 60)


if __name__ == "__main__":
    main()
