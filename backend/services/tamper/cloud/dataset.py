"""
dataset.py — PyTorch Dataset for multi-head tamper detection training.

Location: backend/services/tamper/cloud/

Reads pre-cached ELA images from normalized/ela_cache/ and labels from labels.csv.
Returns (tensor, labels_dict) where labels_dict has keys:
  photo_replacement, text_manipulation, compression_anomaly
Each value is 0, 1, or -1 (masked — don't compute loss for this head).
"""

import os
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiHeadTamperDataset(Dataset):
    """
    Dataset for multi-head tamper detection.
    
    Reads pre-cached ELA images and returns tensors with multi-label targets.
    Labels with value -1 are masked during loss computation.
    """

    def __init__(self, labels_csv, ela_cache_dir, transform=None):
        """
        Args:
            labels_csv: Path to labels.csv
            ela_cache_dir: Path to normalized/ela_cache/
            transform: torchvision transforms to apply
        """
        self.ela_cache_dir = Path(ela_cache_dir)
        self.transform = transform

        # Load labels
        self.samples = []
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                # ELA cache always saves as .jpg regardless of original format
                ela_filename = Path(filename).stem + ".jpg"
                ela_path = self.ela_cache_dir / ela_filename

                if ela_path.exists():
                    self.samples.append({
                        "ela_path": str(ela_path),
                        "photo_replacement": int(row["photo_replacement"]),
                        "text_manipulation": int(row["text_manipulation"]),
                        "compression_anomaly": int(row["compression_anomaly"]),
                        "source": row["source"],
                    })

        print(f"  Dataset loaded: {len(self.samples)} samples with cached ELA")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load pre-computed ELA image
        try:
            ela_img = Image.open(sample["ela_path"]).convert("RGB")
        except Exception:
            # Fallback: black image
            ela_img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            ela_tensor = self.transform(ela_img)
        else:
            ela_tensor = transforms.ToTensor()(ela_img)

        ela_img.close()

        # Labels: each is 0, 1, or -1
        labels = {
            "photo_replacement": sample["photo_replacement"],
            "text_manipulation": sample["text_manipulation"],
            "compression_anomaly": sample["compression_anomaly"],
        }

        return ela_tensor, labels

    def get_source_distribution(self):
        """Returns a dict of source_dataset -> count."""
        from collections import Counter
        return Counter(s["source"] for s in self.samples)

    def get_label_distribution(self):
        """Returns per-head label distributions."""
        heads = ["photo_replacement", "text_manipulation", "compression_anomaly"]
        dist = {}
        for head in heads:
            counts = {0: 0, 1: 0, -1: 0}
            for s in self.samples:
                counts[s[head]] += 1
            dist[head] = counts
        return dist


def collate_multihead(batch):
    """
    Custom collate function that properly batches multi-head labels.
    
    Returns:
        tensors: (B, 3, H, W) tensor
        labels: dict with keys photo_replacement, text_manipulation, compression_anomaly
                each is a (B,) LongTensor
    """
    tensors = torch.stack([item[0] for item in batch])
    
    labels = {
        "photo_replacement": torch.tensor([item[1]["photo_replacement"] for item in batch], dtype=torch.long),
        "text_manipulation": torch.tensor([item[1]["text_manipulation"] for item in batch], dtype=torch.long),
        "compression_anomaly": torch.tensor([item[1]["compression_anomaly"] for item in batch], dtype=torch.long),
    }

    return tensors, labels


# Standard transforms matching ImageNet pretrained weights
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
