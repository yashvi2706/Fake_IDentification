"""
normalize_datasets.py — Unified dataset normalizer for Module 3

Location: backend/services/tamper/cloud/

Reads all 5 raw dataset directories and produces:
  normalized/
  ├── images/        # All images renamed to {dataset}_{idx}.ext
  └── labels.csv     # filename, photo_replacement, text_manipulation, compression_anomaly, source

Label convention:
   0 = authentic / negative
   1 = tampered / positive
  -1 = MASKED (don't compute loss for this head on this sample)
"""

import os
import sys
import csv
import glob
import shutil
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / "raw"
NORM_DIR = SCRIPT_DIR / "normalized"
IMG_DIR = NORM_DIR / "images"
LABELS_CSV = NORM_DIR / "labels.csv"

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(directory, recursive=True):
    """Find all image files in a directory."""
    images = []
    pattern = "**/*" if recursive else "*"
    for ext in IMG_EXTS:
        images.extend(Path(directory).glob(f"{pattern}{ext}"))
        images.extend(Path(directory).glob(f"{pattern}{ext.upper()}"))
    return sorted(set(images))


def copy_image(src, dst_name):
    """Copy an image to the normalized directory with a new name."""
    ext = src.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".jpg"
    dst = IMG_DIR / f"{dst_name}{ext}"
    shutil.copy2(str(src), str(dst))
    return dst.name


def normalize_casia():
    """
    CASIA v2 — Generic forgery dataset.
    Structure: typically has Au (authentic) and Tp (tampered) directories.
    Maps to ALL 3 heads (generic signal).
    """
    casia_dir = RAW_DIR / "casia"
    if not casia_dir.exists():
        print("  ⚠ CASIA directory not found, skipping")
        return []

    rows = []
    idx = 0

    # Find authentic images (directory names vary: Au, Authentic, au, authentic, etc.)
    auth_dirs = []
    tamp_dirs = []
    for d in casia_dir.rglob("*"):
        if d.is_dir():
            name_lower = d.name.lower()
            if name_lower in ("au", "authentic", "real", "original"):
                auth_dirs.append(d)
            elif name_lower in ("tp", "tampered", "fake", "forged", "spliced", "copymove"):
                tamp_dirs.append(d)

    # If standard dirs not found, try top-level structure
    if not auth_dirs and not tamp_dirs:
        # Some CASIA Kaggle versions have CASIA2/Au and CASIA2/Tp
        for d in casia_dir.rglob("*"):
            if d.is_dir():
                if "au" in d.name.lower() and len(d.name) <= 15:
                    auth_dirs.append(d)
                elif "tp" in d.name.lower() and len(d.name) <= 15:
                    tamp_dirs.append(d)

    # Collect authentic
    for adir in auth_dirs:
        for img in find_images(adir, recursive=False):
            fname = copy_image(img, f"casia_auth_{idx}")
            rows.append((fname, 0, 0, 0, "casia"))
            idx += 1

    # Collect tampered — generic forgery maps to all 3 heads
    for tdir in tamp_dirs:
        for img in find_images(tdir, recursive=False):
            fname = copy_image(img, f"casia_tamp_{idx}")
            rows.append((fname, 1, 1, 1, "casia"))
            idx += 1

    print(f"  CASIA: {idx} images ({sum(1 for r in rows if r[1]==0)} auth, {sum(1 for r in rows if r[1]==1)} tamp)")
    return rows


def normalize_fantasyid():
    """
    FantasyID — ID-specific face-swap and text-inpaint attacks.
    Structure varies; look for train/test with subfolders indicating attack type.
    face-swap → photo_replacement=1, text_manipulation=0, compression=-1 (MASK)
    text-inpaint → photo_replacement=0, text_manipulation=1, compression=-1 (MASK)
    bona-fide → all 0s
    """
    fid_dir = RAW_DIR / "fantasyid"
    if not fid_dir.exists():
        print("  ⚠ FantasyID directory not found, skipping")
        return []

    rows = []
    idx = 0

    # Scan for bona-fide / genuine images
    bonafide_keywords = ["bonafide", "bona_fide", "genuine", "real", "original", "authentic"]
    face_swap_keywords = ["faceswap", "face_swap", "face-swap", "swap"]
    text_inpaint_keywords = ["textinpaint", "text_inpaint", "text-inpaint", "inpaint", "text"]
    tampered_generic_keywords = ["tampered", "forged", "fake", "manipulated", "attack"]

    all_images = find_images(fid_dir)
    
    for img in all_images:
        path_lower = str(img).lower()
        
        if any(kw in path_lower for kw in face_swap_keywords):
            fname = copy_image(img, f"fid_faceswap_{idx}")
            rows.append((fname, 1, 0, -1, "fantasyid"))
            idx += 1
        elif any(kw in path_lower for kw in text_inpaint_keywords):
            fname = copy_image(img, f"fid_textinpaint_{idx}")
            rows.append((fname, 0, 1, -1, "fantasyid"))
            idx += 1
        elif any(kw in path_lower for kw in bonafide_keywords):
            fname = copy_image(img, f"fid_bonafide_{idx}")
            rows.append((fname, 0, 0, 0, "fantasyid"))
            idx += 1
        elif any(kw in path_lower for kw in tampered_generic_keywords):
            # Generic tamper — contribute to both photo and text heads
            fname = copy_image(img, f"fid_tampered_{idx}")
            rows.append((fname, 1, 1, -1, "fantasyid"))
            idx += 1
        else:
            # Unknown category — treat as bona-fide (conservative)
            fname = copy_image(img, f"fid_other_{idx}")
            rows.append((fname, 0, 0, 0, "fantasyid"))
            idx += 1

    print(f"  FantasyID: {idx} images")
    return rows


def normalize_docxpand():
    """
    DocXPand-25k — ALL authentic synthetic documents.
    No forgeries. Label 0 across all heads.
    Purpose: reduce false positives on clean docs.
    """
    dx_dir = RAW_DIR / "docxpand"
    if not dx_dir.exists():
        print("  ⚠ DocXPand directory not found, skipping")
        return []

    rows = []
    idx = 0

    all_images = find_images(dx_dir)
    for img in all_images:
        fname = copy_image(img, f"docxpand_{idx}")
        rows.append((fname, 0, 0, 0, "docxpand"))
        idx += 1

    print(f"  DocXPand: {idx} images (all authentic)")
    return rows


def normalize_sidtd():
    """
    SIDTD — Synthetic ID Document Tampering.
    Contains tampered and authentic variants.
    Tampered → photo_replacement=1, text_manipulation=1, compression=-1 (MASK)
    Authentic → all 0s
    """
    sidtd_dir = RAW_DIR / "sidtd"
    if not sidtd_dir.exists():
        print("  ⚠ SIDTD directory not found, skipping")
        return []

    rows = []
    idx = 0

    all_images = find_images(sidtd_dir)
    
    tampered_keywords = ["tampered", "forged", "fake", "manipulated", "attack", "tp", "altered"]
    authentic_keywords = ["authentic", "genuine", "real", "original", "au", "bonafide"]

    for img in all_images:
        path_lower = str(img).lower()
        parent_lower = img.parent.name.lower()

        if any(kw in path_lower for kw in tampered_keywords) or any(kw == parent_lower for kw in ["tp", "tampered", "fake", "forged"]):
            fname = copy_image(img, f"sidtd_tamp_{idx}")
            rows.append((fname, 1, 1, -1, "sidtd"))
            idx += 1
        elif any(kw in path_lower for kw in authentic_keywords) or any(kw == parent_lower for kw in ["au", "authentic", "real", "original"]):
            fname = copy_image(img, f"sidtd_auth_{idx}")
            rows.append((fname, 0, 0, 0, "sidtd"))
            idx += 1
        else:
            # Default: treat as authentic (conservative for unknown structure)
            fname = copy_image(img, f"sidtd_unk_{idx}")
            rows.append((fname, 0, 0, 0, "sidtd"))
            idx += 1

    print(f"  SIDTD: {idx} images")
    return rows


def normalize_recaptured_bid():
    """
    Recaptured Identity Documents (BID).
    Screen/print recaptures → compression_anomaly=1, others=-1 (MASK)
    Genuine → all 0s
    """
    bid_dir = RAW_DIR / "recaptured_bid"
    if not bid_dir.exists():
        print("  ⚠ Recaptured BID directory not found, skipping")
        return []

    rows = []
    idx = 0

    all_images = find_images(bid_dir)
    
    recapture_keywords = ["recaptured", "recapture", "screen", "print", "copy", "fake", "attack"]
    genuine_keywords = ["genuine", "original", "real", "authentic", "bonafide"]

    for img in all_images:
        path_lower = str(img).lower()
        parent_lower = img.parent.name.lower()

        if any(kw in path_lower for kw in recapture_keywords) or parent_lower in ["recaptured", "fake", "screen", "print"]:
            fname = copy_image(img, f"bid_recap_{idx}")
            rows.append((fname, -1, -1, 1, "recaptured_bid"))
            idx += 1
        elif any(kw in path_lower for kw in genuine_keywords) or parent_lower in ["genuine", "original", "real"]:
            fname = copy_image(img, f"bid_genuine_{idx}")
            rows.append((fname, 0, 0, 0, "recaptured_bid"))
            idx += 1
        else:
            # Unknown — try to infer from directory structure
            device_keywords = ["iphone", "samsung", "pixel", "huawei", "camera"]
            if any(kw in parent_lower for kw in device_keywords):
                fname = copy_image(img, f"bid_recap_{idx}")
                rows.append((fname, -1, -1, 1, "recaptured_bid"))
            else:
                fname = copy_image(img, f"bid_unk_{idx}")
                rows.append((fname, 0, 0, 0, "recaptured_bid"))
            idx += 1

    print(f"  Recaptured BID: {idx} images")
    return rows


def main():
    print("============================================")
    print("  Normalizing datasets into unified format")
    print("============================================")
    print()

    # Create output directories
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # Process each dataset
    print("Processing CASIA v2...")
    all_rows.extend(normalize_casia())
    print()

    print("Processing FantasyID...")
    all_rows.extend(normalize_fantasyid())
    print()

    print("Processing DocXPand-25k...")
    all_rows.extend(normalize_docxpand())
    print()

    print("Processing SIDTD...")
    all_rows.extend(normalize_sidtd())
    print()

    print("Processing Recaptured BID...")
    all_rows.extend(normalize_recaptured_bid())
    print()

    # Write labels.csv
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "photo_replacement", "text_manipulation", "compression_anomaly", "source"])
        writer.writerows(all_rows)

    # Summary statistics
    sources = Counter(r[4] for r in all_rows)
    print("============================================")
    print(f"  Total images: {len(all_rows)}")
    print(f"  Labels CSV: {LABELS_CSV}")
    print()
    print("  Per-dataset breakdown:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")
    print()

    # Per-head statistics (excluding masked)
    for head_idx, head_name in [(1, "photo_replacement"), (2, "text_manipulation"), (3, "compression_anomaly")]:
        positives = sum(1 for r in all_rows if r[head_idx] == 1)
        negatives = sum(1 for r in all_rows if r[head_idx] == 0)
        masked = sum(1 for r in all_rows if r[head_idx] == -1)
        print(f"  {head_name}: {positives} pos / {negatives} neg / {masked} masked")

    print("============================================")


if __name__ == "__main__":
    main()
