"""
Lightweight tamper-detection service — zero ML dependencies.

Replaces the previous PyTorch / TruFor implementation with three
pure-OpenCV / numpy heuristics that run inside 512 MB of RAM:

1. **ELA (Error Level Analysis)**
   Re-compress the image at a known quality and compare pixel-level
   differences. Edited regions typically show higher residuals.

2. **Copy-move detection (block-DCT)**
   Divide the image into 16×16 blocks, hash their DCT coefficients,
   and flag suspicious duplicate-block pairs.

3. **Metadata anomalies**
   EXIF creation-vs-modification date mismatches, missing camera make,
   suspiciously low/high compression ratio.

The three signals are fused into a final score [0, 100].
`analyze_tampering()` never raises — all exceptions are caught and
result in a degraded-gracefully response.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import exifread
    HAS_EXIFREAD = True
except Exception:
    HAS_EXIFREAD = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────
ELA_QUALITY = 92          # JPEG re-compress quality for ELA
ELA_SCALE = 10            # amplification for visualisation (not used in score)
ELA_HIGH_THRESH = 18.0    # mean residual above this → high tampering signal
ELA_MED_THRESH = 10.0     # mean residual above this → moderate signal

COPY_BLOCK = 16           # block size for copy-move DCT hashing
COPY_HIGH = 12            # ≥ this many duplicate pairs → high signal

META_MAX_POINTS = 20      # cap metadata sub-score

FUSION = {                # weights must sum to 1.0
    "ela": 0.50,
    "copy_move": 0.30,
    "metadata": 0.20,
}
TOTAL_CAP = 100


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_tampering(image_path: str) -> Dict[str, Any]:
    """
    Run all tamper-detection heuristics and return a unified result dict.

    The returned dict is always safe (never raises).

    Keys:
        score          int   0–100, higher = more suspicious
        level          str   "LOW" | "MEDIUM" | "HIGH"
        signals        list  per-heuristic breakdowns
        error          str?  present only on unexpected failure
    """
    try:
        return _analyze(image_path)
    except Exception as exc:
        logger.error("analyze_tampering unexpected error: %s", exc, exc_info=True)
        return {
            "score": 0,
            "level": "LOW",
            "signals": [],
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _analyze(image_path: str) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    signals: List[Dict[str, Any]] = []

    # 1. ELA
    ela_score, ela_detail = _ela_score(image_path)
    signals.append({"name": "ELA", "score": ela_score, "detail": ela_detail})

    # 2. Copy-move
    cm_score, cm_detail = _copy_move_score(image_path)
    signals.append({"name": "copy_move", "score": cm_score, "detail": cm_detail})

    # 3. Metadata
    meta_score, meta_detail = _metadata_score(image_path)
    signals.append({"name": "metadata", "score": meta_score, "detail": meta_detail})

    # Fused score
    raw = (
        ela_score * FUSION["ela"]
        + cm_score * FUSION["copy_move"]
        + meta_score * FUSION["metadata"]
    )
    score = int(min(TOTAL_CAP, round(raw)))

    if score < 30:
        level = "LOW"
    elif score < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        # Schema-required fields (used by frontend)
        "score": float(score),
        "suspicious": score >= SUSPICIOUS_THRESHOLD,
        "photo_replacement": cm_score >= 50.0,
        "text_manipulation": ela_score >= ELA_HIGH_THRESH * 3,  # very high ELA = likely text edit
        "metadata_anomaly": meta_score >= 30.0,
        "compression_anomaly": ela_score >= ELA_MED_THRESH * 2,
        "indicators": [s["detail"] for s in signals if s["score"] > 0],
        # Extra context (ignored by schema but logged server-side)
        "level": level,
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic 1 — Error Level Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _ela_score(image_path: str) -> Tuple[float, str]:
    """
    Compute ELA score [0, 100].

    Algorithm:
      • Load original image as PIL.
      • Re-save to an in-memory JPEG buffer at ELA_QUALITY.
      • Reload re-saved image.
      • Compute per-channel absolute difference.
      • Mean of difference as tamper indicator.
    """
    try:
        original = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr = np.array(original, dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)
        diff = np.abs(orig_arr - recomp_arr)
        mean_diff = float(diff.mean())

        if mean_diff >= ELA_HIGH_THRESH:
            score = min(100.0, 60.0 + (mean_diff - ELA_HIGH_THRESH) * 3)
            detail = f"High residual ({mean_diff:.1f}) — likely edited"
        elif mean_diff >= ELA_MED_THRESH:
            ratio = (mean_diff - ELA_MED_THRESH) / (ELA_HIGH_THRESH - ELA_MED_THRESH)
            score = 20.0 + ratio * 40.0
            detail = f"Moderate residual ({mean_diff:.1f})"
        else:
            score = mean_diff / ELA_MED_THRESH * 20.0
            detail = f"Low residual ({mean_diff:.1f}) — appears unedited"

        return round(score, 2), detail

    except Exception as exc:
        logger.debug("ELA failed (non-fatal): %s", exc)
        return 0.0, f"ELA unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic 2 — Copy-Move (block DCT hashing)
# ─────────────────────────────────────────────────────────────────────────────

def _copy_move_score(image_path: str) -> Tuple[float, str]:
    """
    Detect duplicate image blocks via DCT coefficient hashing.
    Score [0, 100].
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, "Could not load image"

        # Resize to max 512 px wide for speed
        h, w = img.shape
        if w > 512:
            scale = 512 / w
            img = cv2.resize(img, (512, int(h * scale)))
            h, w = img.shape

        b = COPY_BLOCK
        blocks: Dict[bytes, int] = {}
        duplicates = 0

        for y in range(0, h - b, b):
            for x in range(0, w - b, b):
                block = img[y : y + b, x : x + b].astype(np.float32)
                dct = cv2.dct(block)
                # Use top-left 4×4 AC coefficients as hash key
                key = dct[:4, :4].tobytes()
                if key in blocks:
                    duplicates += 1
                else:
                    blocks[key] = 1

        if duplicates >= COPY_HIGH:
            score = min(100.0, 50.0 + duplicates * 2)
            detail = f"{duplicates} duplicate block(s) — suspicious"
        elif duplicates > 0:
            score = duplicates / COPY_HIGH * 50.0
            detail = f"{duplicates} duplicate block(s) — low signal"
        else:
            score = 0.0
            detail = "No duplicate blocks detected"

        return round(score, 2), detail

    except Exception as exc:
        logger.debug("Copy-move detection failed (non-fatal): %s", exc)
        return 0.0, f"Copy-move unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic 3 — Metadata / EXIF anomalies
# ─────────────────────────────────────────────────────────────────────────────

def _metadata_score(image_path: str) -> Tuple[float, str]:
    """
    Inspect EXIF metadata for anomalies. Score [0, META_MAX_POINTS].
    """
    points = 0.0
    notes: List[str] = []

    # PIL-level checks (always available)
    try:
        img = Image.open(image_path)
        exif_data = img._getexif() if hasattr(img, "_getexif") else None

        if img.format == "JPEG":
            # Missing EXIF entirely in a JPEG is unusual for camera photos
            if exif_data is None:
                points += 5
                notes.append("No EXIF data in JPEG")
    except Exception:
        pass

    # exifread checks
    if HAS_EXIFREAD:
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False, stop_tag="UNDEF")

            make = tags.get("Image Make")
            model = tags.get("Image Model")
            software = tags.get("Image Software")

            if not make and not model:
                points += 5
                notes.append("Camera make/model missing")

            if software:
                sw = str(software).lower()
                edit_keywords = [
                    "photoshop", "gimp", "lightroom", "affinity",
                    "paint", "snapseed", "facetune", "pixlr",
                ]
                for kw in edit_keywords:
                    if kw in sw:
                        points += 8
                        notes.append(f"Editing software detected: {software}")
                        break

            # Date created vs modified mismatch
            dt_orig = tags.get("EXIF DateTimeOriginal")
            dt_digit = tags.get("EXIF DateTimeDigitized")
            if dt_orig and dt_digit and str(dt_orig) != str(dt_digit):
                points += 4
                notes.append("Creation/digitized timestamps differ")

        except Exception as exc:
            logger.debug("EXIF read failed (non-fatal): %s", exc)

    score = min(float(META_MAX_POINTS), points) / META_MAX_POINTS * 100.0
    detail = "; ".join(notes) if notes else "No metadata anomalies"
    return round(score, 2), detail
