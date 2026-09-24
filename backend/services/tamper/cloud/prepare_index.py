"""Prepare source-disjoint index CSV from user-supplied dataset directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_source_spec(spec: str) -> Tuple[str, str, int, int]:
    # format: source_name:/abs/path:photo_label:doc_label
    parts = spec.split(":")
    if len(parts) < 4:
        raise ValueError(f"Invalid --source format: {spec}")
    source = parts[0]
    path = ":".join(parts[1:-2])
    p_label = int(parts[-2])
    d_label = int(parts[-1])
    return source, path, p_label, d_label


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def image_quick_hash(path: Path) -> str:
    # near-duplicate guard (cheap): perceptual-ish signature from tiny grayscale
    try:
        img = Image.open(path).convert("L").resize((16, 16), Image.BILINEAR)
        arr = bytes(img.tobytes())
        return hashlib.sha1(arr).hexdigest()
    except Exception:
        return "bad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="source_name:/path:photo_label:doc_label")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = []
    seen_hash = set()

    for spec in args.source:
        source, path, photo_label, doc_label = parse_source_spec(spec)
        root = Path(path)
        if not root.exists():
            print(f"[WARN] source path missing: {root}")
            continue

        src_count = 0
        for p in iter_images(root):
            h = image_quick_hash(p)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            rows.append(
                {
                    "path": str(p.resolve()),
                    "source": source,
                    "label_photo": photo_label,
                    "label_document": doc_label,
                }
            )
            src_count += 1
        print(f"[INFO] {source}: kept {src_count} unique images")

    # source-disjoint split: whole sources go to train/val/test groups
    sources = sorted(set(r["source"] for r in rows))
    random.shuffle(sources)

    n = len(sources)
    n_test = max(1, int(round(n * args.test_ratio))) if n >= 3 else 1
    n_val = max(1, int(round(n * args.val_ratio))) if n >= 3 else 1

    test_sources = set(sources[:n_test])
    val_sources = set(sources[n_test : n_test + n_val])
    train_sources = set(sources[n_test + n_val :])
    if not train_sources:
        train_sources = set(sources[n_test:])
        val_sources = set()

    for r in rows:
        if r["source"] in test_sources:
            r["split"] = "test"
        elif r["source"] in val_sources:
            r["split"] = "val"
        else:
            r["split"] = "train"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "split", "label_photo", "label_document"])
        w.writeheader()
        w.writerows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print(f"[DONE] wrote {len(rows)} rows to {out_path}")
    print(f"[SPLIT] train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[SOURCES] train={sorted(train_sources)} val={sorted(val_sources)} test={sorted(test_sources)}")


if __name__ == "__main__":
    main()
