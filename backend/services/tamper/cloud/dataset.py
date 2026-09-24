"""Dataset and synthetic SBI-style augmentation for tamper training."""

from __future__ import annotations

import csv
import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import cv2

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class IndexRow:
    path: str
    source: str
    split: str
    label_photo: int
    label_document: int


def load_index_csv(path: str) -> List[IndexRow]:
    rows: List[IndexRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                IndexRow(
                    path=r["path"],
                    source=r["source"],
                    split=r["split"],
                    label_photo=int(r.get("label_photo", -1)),
                    label_document=int(r.get("label_document", 0)),
                )
            )
    return rows


def make_ela(img_rgb: Image.Image, quality: int = 90, scale: int = 20, size: int = 224) -> Image.Image:
    try:
        buf = io.BytesIO()
        img_rgb.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")
        diff = np.abs(np.asarray(img_rgb, np.int16) - np.asarray(comp, np.int16))
        ela = np.clip(diff * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(ela).resize((size, size), Image.BILINEAR)
    except Exception:
        return Image.new("RGB", (size, size), (0, 0, 0))


def _detect_face_box(img_rgb: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    if not HAS_CV2:
        return None
    try:
        detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        if detector.empty():
            return None
        arr = np.asarray(img_rgb)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return int(x), int(y), int(x + w), int(y + h)
    except Exception:
        return None


def _random_box(w: int, h: int) -> Tuple[int, int, int, int]:
    bw = random.randint(max(24, w // 8), max(30, w // 3))
    bh = random.randint(max(24, h // 8), max(30, h // 3))
    x1 = random.randint(0, max(0, w - bw))
    y1 = random.randint(0, max(0, h - bh))
    return x1, y1, x1 + bw, y1 + bh


def _jpeg_reencode(img: Image.Image) -> Image.Image:
    q = random.randint(65, 98)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def make_sbi_tamper(img_rgb: Image.Image) -> Tuple[Image.Image, bool]:
    """
    On-the-fly synthetic self-blending tamper generation.
    Returns tampered image and whether the manipulated region was face-centric.
    """
    w, h = img_rgb.size
    if w < 40 or h < 40:
        return img_rgb, False

    box = _detect_face_box(img_rgb)
    face_like = box is not None
    if box is None:
        box = _random_box(w, h)

    x1, y1, x2, y2 = box
    region = img_rgb.crop((x1, y1, x2, y2))

    # aggressive but realistic perturbations
    if random.random() < 0.7:
        region = ImageEnhance.Color(region).enhance(random.uniform(0.75, 1.35))
    if random.random() < 0.7:
        region = ImageEnhance.Brightness(region).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.6:
        region = ImageEnhance.Contrast(region).enhance(random.uniform(0.8, 1.3))
    if random.random() < 0.4:
        region = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    dx = int((x2 - x1) * random.uniform(-0.12, 0.12))
    dy = int((y2 - y1) * random.uniform(-0.12, 0.12))
    nx1 = max(0, min(w - 1, x1 + dx))
    ny1 = max(0, min(h - 1, y1 + dy))
    nx2 = min(w, nx1 + (x2 - x1))
    ny2 = min(h, ny1 + (y2 - y1))

    mask = Image.new("L", region.size, 0)
    if HAS_CV2:
        m = np.zeros((region.size[1], region.size[0]), dtype=np.uint8)
        cv2.ellipse(
            m,
            (region.size[0] // 2, region.size[1] // 2),
            (max(6, region.size[0] // 2 - 3), max(6, region.size[1] // 2 - 3)),
            0,
            0,
            360,
            255,
            -1,
        )
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=random.uniform(2.0, 4.5))
        mask = Image.fromarray(m)
    else:
        mask = Image.new("L", region.size, color=180)

    out = img_rgb.copy()
    paste_w = max(1, nx2 - nx1)
    paste_h = max(1, ny2 - ny1)
    region = region.resize((paste_w, paste_h), Image.BILINEAR)
    mask = mask.resize((paste_w, paste_h), Image.BILINEAR)
    out.paste(region, (nx1, ny1), mask)

    if random.random() < 0.8:
        out = _jpeg_reencode(out)
    if random.random() < 0.5:
        out = out.resize((random.randint(180, 512), random.randint(180, 512)), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    return out, face_like


class TamperTrainDataset(Dataset):
    def __init__(self, index_csv: str, split: str, image_size: int = 224, tamper_prob: float = 0.5):
        all_rows = load_index_csv(index_csv)
        self.rows = [r for r in all_rows if r.split == split]
        self.image_size = image_size
        self.tamper_prob = tamper_prob

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        try:
            img = Image.open(row.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        # Synthetic tamper generation from real images on the fly
        photo_label = row.label_photo
        doc_label = row.label_document

        if row.label_document == 0 and random.random() < self.tamper_prob:
            img, face_like = make_sbi_tamper(img)
            doc_label = 1
            photo_label = 1 if face_like else -1

        ela = make_ela(img, size=self.image_size)
        rgb_t = self.to_tensor(img)
        ela_t = self.to_tensor(ela)
        x6 = torch.cat([rgb_t, ela_t], dim=0)

        labels = {
            "photo_replacement": torch.tensor(int(photo_label), dtype=torch.long),
            "document_tamper": torch.tensor(int(doc_label), dtype=torch.long),
        }
        return x6, labels, row.source


class TamperEvalDataset(Dataset):
    def __init__(self, index_csv: str, split: str, image_size: int = 224):
        all_rows = load_index_csv(index_csv)
        self.rows = [r for r in all_rows if r.split == split]
        self.image_size = image_size
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        try:
            img = Image.open(row.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        ela = make_ela(img, size=self.image_size)
        rgb_t = self.to_tensor(img)
        ela_t = self.to_tensor(ela)
        x6 = torch.cat([rgb_t, ela_t], dim=0)
        labels = {
            "photo_replacement": torch.tensor(int(row.label_photo), dtype=torch.long),
            "document_tamper": torch.tensor(int(row.label_document), dtype=torch.long),
        }
        return x6, labels, row.source


def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    labels = {
        "photo_replacement": torch.stack([b[1]["photo_replacement"] for b in batch], dim=0),
        "document_tamper": torch.stack([b[1]["document_tamper"] for b in batch], dim=0),
    }
    sources = [b[2] for b in batch]
    return xs, labels, sources
