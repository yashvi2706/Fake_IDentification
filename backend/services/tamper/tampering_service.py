"""
tampering_service.py — Standalone tamper detection module.

Integration contract:
    from services.tampering_service import analyze_tampering
    result = analyze_tampering(image_bytes)

This module is FULLY INDEPENDENT — no imports from OCR, validation, face, or risk.

Models (all loaded once at import time, kept in memory):
    * MultiHeadResNet18 (ELA-based)  -> photo_replacement head (only trained head)
    * TruFor (CVPR 2023, confcmx)    -> text_manipulation (global integrity score)
Both fall back gracefully: if a model is missing the corresponding sub-score is 0
and the response says so via details["trufor_available"] / status helpers.

Scoring (deterministic, capped at 100):
    photo_replacement:   0-65  (CNN photo head)
    text_manipulation:   0-65  (TruFor sigmoid(det) * 65)
    metadata_anomaly:    0-10  (EXIF heuristic)
    compression_anomaly: 0-15  (ELA max_diff heuristic)
    ────────────────────────
    Raw total:           0-155 (hard cap: 100)

Threshold: suspicious = True when score >= 60

Environment variables (all optional):
    TRUFOR_HOME          Path to the TruFor repo root (contains test_docker/)
    TRUFOR_MAX_SIDE      Downscale long side to N px before TruFor. 0 / unset = native
                         resolution (default, best accuracy).
    TAMPER_DEVICE        "cpu" (default) or e.g. "cuda:0"
"""

import os
import io
import sys
import uuid
import types
import tempfile
import threading
import traceback
import contextlib

import numpy as np  # hard requirement — used in compute_ela to match training pipeline
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False


# =============================================================================
# Configuration
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))

# Score budget
PHOTO_MAX_POINTS = 65
TEXT_MAX_POINTS = 65          # TruFor
METADATA_MAX_POINTS = 10
COMPRESSION_MAX_POINTS = 15
TOTAL_CAP = 100
SUSPICIOUS_THRESHOLD = 60


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# 0 = run TruFor at native resolution (no quality loss). Only set this if you
# actually need to bound latency / memory.
TRUFOR_MAX_SIDE = _env_int("TRUFOR_MAX_SIDE", 0)

# Only used as an emergency retry if native-resolution inference runs out of memory.
_TRUFOR_OOM_RETRY_SIDE = 2048


def _resolve_device():
    name = os.environ.get("TAMPER_DEVICE", "cpu")
    try:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("[tampering_service] CUDA requested but unavailable — using CPU")
            return torch.device("cpu")
        return dev
    except Exception:
        return torch.device("cpu")


_device = _resolve_device()


# =============================================================================
# Model Definition (inference-only copy — matches cloud/model.py)
# =============================================================================

class MultiHeadResNet18(nn.Module):
    """Multi-head ResNet18 for tamper detection inference."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        feat_dim = 512
        self.head_photo = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2),
        )
        self.head_text = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2),
        )
        # NOTE: must be head_comp, NOT head_compression — checkpoint keys are head_comp.*
        self.head_comp = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2),
        )

    def forward(self, x):
        features = self.features(x).flatten(1)
        return {
            "photo_replacement": self.head_photo(features),
            "text_manipulation": self.head_text(features),
            "compression_anomaly": self.head_comp(features),
        }


# =============================================================================
# Global model loading — ResNet18 (photo head)
# =============================================================================

_model = None
_model_loaded = False

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Checkpoint search paths (in order of priority)
_CHECKPOINT_NAMES = [
    "tamper_multihead_resnet18.pth",   # Multi-head model (new)
    "tamper_resnet18.pth",              # Legacy single-head (fallback)
]

_CHECKPOINT_DIRS = [
    _HERE,                                                   # Same dir as this file
    os.path.join(_HERE, "cloud", "checkpoints"),             # cloud/ subdir
    "/app/services/tamper",                                  # Docker mount
    "/app/services/tamper/cloud/checkpoints",                # Docker cloud subdir
]


def _load_model():
    """Attempt to load the ResNet18 checkpoint."""
    global _model, _model_loaded

    for ckpt_dir in _CHECKPOINT_DIRS:
        for ckpt_name in _CHECKPOINT_NAMES:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                try:
                    if "multihead" in ckpt_name:
                        m = MultiHeadResNet18()
                        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
                        m.to(_device).eval()
                        _model = m
                        _model_loaded = True
                        print(f"[tampering_service] Multi-head model loaded from {ckpt_path}")
                        return True
                    else:
                        # Legacy single-head model — load but flag it
                        legacy_model = models.resnet18(weights=None)
                        legacy_model.fc = nn.Linear(legacy_model.fc.in_features, 2)
                        legacy_model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
                        legacy_model.to(_device).eval()
                        _model = legacy_model
                        _model_loaded = True
                        print(f"[tampering_service] Legacy single-head model loaded from {ckpt_path}")
                        return True
                except Exception as e:
                    print(f"[tampering_service] Failed to load {ckpt_path}: {e}")
                    continue

    print("[tampering_service] No ResNet18 checkpoint found — photo head disabled")
    return False


_load_model()


# =============================================================================
# Global model loading — TruFor (text manipulation)
# =============================================================================

_trufor_model = None
_trufor_device = _device
_trufor_lock = threading.Lock()   # serialize inference: native-res passes are memory heavy


def _find_trufor_paths():
    """
    Locate TruFor's test_docker/src dir and weights file.
    Returns (src_dir, weights_path) or None.
    """
    roots = []
    env_root = os.environ.get("TRUFOR_HOME")
    if env_root:
        roots.append(env_root)
    roots += [
        os.path.join(_HERE, "TruFor"),
        os.path.join(os.path.dirname(_HERE), "TruFor"),
        os.path.join(os.getcwd(), "TruFor"),
        "/app/services/tamper/TruFor",
        "/app/TruFor",
    ]
    for root in roots:
        src = os.path.join(root, "test_docker", "src")
        weights = os.path.join(root, "test_docker", "weights", "trufor.pth.tar")
        if os.path.isdir(src) and os.path.isfile(weights):
            return src, weights
    return None


@contextlib.contextmanager
def _trufor_import_context(src_dir):
    """
    TruFor's src uses generic top-level names (config, models, data_core, ...).
    Put its src dir on sys.path only while importing, hide any same-named modules
    already imported by the host project, and put everything back afterwards so
    TruFor can never shadow (or be shadowed by) the rest of the application.
    """
    top_names = set()
    for entry in os.listdir(src_dir):
        full = os.path.join(src_dir, entry)
        if os.path.isdir(full):
            top_names.add(entry)
        elif entry.endswith(".py"):
            top_names.add(entry[:-3])

    def _is_trufor_name(mod_name):
        return mod_name.split(".")[0] in top_names

    shadowed = {n: m for n, m in sys.modules.items() if _is_trufor_name(n)}
    for n in shadowed:
        del sys.modules[n]

    sys.path.insert(0, src_dir)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(src_dir)
        for n in [n for n in sys.modules if _is_trufor_name(n)]:
            del sys.modules[n]
        sys.modules.update(shadowed)


def _load_trufor():
    """Load TruFor (confcmx) into the global _trufor_model. Never raises."""
    global _trufor_model

    try:
        import timm   # noqa: F401  (dependency check only)
        import yacs   # noqa: F401  (dependency check only)
    except ImportError as e:
        print(f"[tampering_service] TruFor dependency missing ({e}) — text head disabled")
        return False

    found = _find_trufor_paths()
    if found is None:
        print("[tampering_service] TruFor src/weights not found — text head disabled")
        return False
    src_dir, weights_path = found

    try:
        with _trufor_import_context(src_dir):
            from config import _C as trufor_cfg
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

            # Official command just runs with defaults + yaml config.
            trufor_cfg.defrost()
            trufor_cfg.merge_from_file(os.path.join(src_dir, "trufor.yaml"))
            if hasattr(trufor_cfg.MODEL, "PRETRAINED"):
                trufor_cfg.MODEL.PRETRAINED = ""
            trufor_cfg.freeze()

            if trufor_cfg.MODEL.NAME != "detconfcmx":
                raise RuntimeError(f"Unexpected TruFor model name: {trufor_cfg.MODEL.NAME}")

            model = confcmx(cfg=trufor_cfg)

            try:
                ckpt = torch.load(weights_path, map_location=_trufor_device, weights_only=True)
            except Exception:
                # Official checkpoint may contain non-tensor objects; file is a trusted local asset.
                ckpt = torch.load(weights_path, map_location=_trufor_device, weights_only=False)

            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state)          # strict: fail loudly on any mismatch
            model.to(_trufor_device).eval()

            # Warm-up so the first real request doesn't pay one-time init cost.
            with torch.no_grad():
                model(torch.zeros(1, 3, 512, 512, device=_trufor_device))

            _trufor_model = model
            print(f"[tampering_service] TruFor loaded from {weights_path}")
            return True

    except Exception as e:
        _trufor_model = None
        print(f"[tampering_service] TruFor unavailable: {e}")
        traceback.print_exc()
        return False


_load_trufor()


def get_model_status():
    """Small helper for health checks / logging."""
    return {
        "resnet18_loaded": _model is not None,
        "resnet18_multihead": isinstance(_model, MultiHeadResNet18),
        "trufor_loaded": _trufor_model is not None,
        "trufor_max_side": TRUFOR_MAX_SIDE or "native",
        "device": str(_device),
    }


# =============================================================================
# ELA (Error Level Analysis)
# =============================================================================

def compute_ela(img_bytes, quality=90, scale=20, size=224):
    """
    Compute ELA matching the training pipeline exactly:
      diff = abs(original - recompressed)  [per-channel int16]
      ela  = clip(diff * scale, 0, 255)    [uint8]
      resize to (size, size) bilinear, then save/reload at q95 JPEG

    The old Brightness.enhance approach saturated early and produced
    a ~5100/max multiplier — totally different from what the CNN trained on.

    Returns:
        ela_image: PIL Image (RGB, 224x224)
        max_diff:  int — single-pixel worst-case difference (0-255)

    Never raises — returns a black image + 0 on failure.
    """
    try:
        original = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")

        diff = np.abs(
            np.asarray(original, np.int16) - np.asarray(compressed, np.int16)
        )
        max_diff = int(diff.max())

        ela_arr = np.clip(diff * scale, 0, 255).astype(np.uint8)
        ela_img = Image.fromarray(ela_arr).resize((size, size), Image.BILINEAR)

        # Round-trip through q95 JPEG exactly as the training cache did
        buf2 = io.BytesIO()
        ela_img.save(buf2, "JPEG", quality=95)
        buf2.seek(0)
        ela_img = Image.open(buf2).convert("RGB")
        ela_img.load()  # force decode before buffer goes out of scope

        return ela_img, max_diff

    except Exception:
        return Image.new("RGB", (224, 224), (0, 0, 0)), 0


# =============================================================================
# EXIF / Metadata Analysis
# =============================================================================

def analyze_exif(img_bytes):
    """
    Analyze EXIF metadata for tampering indicators.

    Returns:
        score: int (0-10)
        indicators: list of human-readable strings
        software: str or None

    Never raises — returns (0, [], None) on failure.
    """
    if not HAS_EXIFREAD:
        return 0, [], None

    try:
        tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
    except Exception:
        return 0, [], None

    score = 0
    indicators = []
    software = None

    # Check for editing software
    software_tag = tags.get("Image Software")
    if software_tag:
        software = str(software_tag)
        suspicious_editors = ["photoshop", "gimp", "paint", "lightroom", "affinity", "pixlr", "canva"]
        if any(ed in software.lower() for ed in suspicious_editors):
            score += 5
            indicators.append(f"Image metadata references editing software ({software})")

    # Check timestamp mismatch
    dt_orig = tags.get("EXIF DateTimeOriginal")
    dt_mod = tags.get("Image DateTime")
    if dt_orig and dt_mod and str(dt_orig) != str(dt_mod):
        score += 3
        indicators.append("Metadata timestamps do not match (DateTimeOriginal vs DateTime)")

    # Check for missing EXIF on JPEG (suspicious for documents — usually stripped)
    if not tags:
        # No EXIF at all — mildly suspicious for a scanned document
        score += 2
        indicators.append("Image contains no EXIF metadata (potentially stripped)")
    elif not dt_orig and not tags.get("EXIF ExifImageWidth"):
        # Minimal EXIF — potentially re-saved
        score += 1

    return min(score, METADATA_MAX_POINTS), indicators, software


# =============================================================================
# CNN Inference (ResNet18, ELA input)
# =============================================================================

def _run_cnn_inference(ela_image):
    """
    Run the CNN model on an ELA image.

    Returns dict of head_name -> probability (0.0 to 1.0).
    Falls back to {all: 0.0} if model not loaded.
    """
    default_scores = {
        "photo_replacement": 0.0,
        "text_manipulation": 0.0,
        "compression_anomaly": 0.0,
    }

    if _model is None:
        return default_scores

    try:
        tensor = _transform(ela_image).unsqueeze(0).to(_device)

        with torch.no_grad():
            if isinstance(_model, MultiHeadResNet18):
                logits = _model(tensor)
                return {
                    head_name: torch.nn.functional.softmax(head_logits, dim=1)[0, 1].item()
                    for head_name, head_logits in logits.items()
                }
            else:
                # Legacy single-head model — only the photo score is consumed downstream
                outputs = _model(tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)[0, 1].item()
                return {
                    "photo_replacement": prob,
                    "text_manipulation": 0.0,
                    "compression_anomaly": 0.0,
                }

    except Exception:
        return default_scores


# =============================================================================
# TruFor Inference (text / content manipulation)
# =============================================================================

def _is_oom(exc):
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def _trufor_forward(img_bytes, max_side):
    """
    One TruFor forward pass. Preprocessing mirrors TruFor's myDataset exactly:
        RGB uint8 -> transpose(2, 0, 1) -> float / 256.0, batch of 1, native size.
    Returns the integrity probability sigmoid(det) as a float in [0, 1].
    Localization / confidence / Noiseprint++ outputs are intentionally discarded.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    rgb = torch.from_numpy(arr).float().div_(256.0).unsqueeze(0).to(_trufor_device)

    found = _find_trufor_paths()
    if not found:
        raise RuntimeError("TruFor paths lost during inference")

    with _trufor_import_context(found[0]):
        with torch.no_grad():
            out = _trufor_model(rgb)

    det = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
    if det is None:
        raise RuntimeError("TruFor returned no detection score")

    return float(torch.sigmoid(det).flatten()[0].item())


def _run_trufor(img_bytes):
    """
    Returns (probability 0-1 or None, downscaled: bool).

    None means TruFor was unavailable or failed — callers must not treat that as "clean".
    Runs at native resolution unless TRUFOR_MAX_SIDE is set. If (and only if) the native
    pass runs out of memory, retries once at a reduced size and reports downscaled=True.
    """
    if _trufor_model is None:
        return None, False

    with _trufor_lock:
        try:
            return _trufor_forward(img_bytes, TRUFOR_MAX_SIDE), bool(TRUFOR_MAX_SIDE)
        except Exception as e:
            if _is_oom(e) and not TRUFOR_MAX_SIDE:
                print("[tampering_service] TruFor OOM at native size — retrying downscaled")
                if _trufor_device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    return _trufor_forward(img_bytes, _TRUFOR_OOM_RETRY_SIDE), True
                except Exception as e2:
                    print(f"[tampering_service] TruFor retry failed: {e2}")
                    return None, False
            print(f"[tampering_service] TruFor inference failed: {e}")
            return None, False


# =============================================================================
# ELA Visualization (optional)
# =============================================================================

def generate_ela_visualization(ela_image):
    """
    Save ELA image as a visualization file.
    Returns path string or None if it fails.

    Never raises.
    """
    try:
        vis_filename = f"ela_{uuid.uuid4().hex[:8]}.png"
        vis_path = os.path.join(tempfile.gettempdir(), vis_filename)
        ela_image.save(vis_path, "PNG")
        return vis_path
    except Exception:
        return None


# =============================================================================
# Main Entry Point
# =============================================================================

def _empty_result(indicators=None):
    return {
        "score": 0,
        "suspicious": False,
        "details": {
            "photo_replacement": 0,
            "text_manipulation": 0,
            "metadata_anomaly": 0,
            "compression_anomaly": 0,
            "trufor_probability": None,
            "trufor_available": False,
            "trufor_downscaled": False,
            "visualization": None,
        },
        "indicators": list(indicators or []),
    }


def analyze_tampering(image_bytes):
    """
    Analyze an image for signs of tampering.

    Args:
        image_bytes: bytes — raw image file content (JPEG, PNG, etc.)

    Returns:
        dict with keys:
            score: int (0-100, deterministic)
            suspicious: bool (True if score >= 60)
            details: dict with sub-scores, TruFor status and optional visualization path
            indicators: list of human-readable strings (empty if clean)

    NEVER raises an exception — returns a safe default on any failure.
    """
    try:
        return _analyze_tampering_impl(image_bytes)
    except Exception as e:
        # Absolute last-resort fallback — pipeline must never crash
        print(f"[tampering_service] Critical error: {e}")
        traceback.print_exc()
        return _empty_result()


def _analyze_tampering_impl(image_bytes):
    """Internal implementation — may raise, caught by analyze_tampering()."""

    # Validate input
    if not image_bytes or len(image_bytes) < 100:
        return _empty_result(["Image too small or empty to analyze"])

    indicators = []

    # ── Step 1: ELA ──────────────────────────────────────────────
    ela_image, ela_max_diff = compute_ela(image_bytes)

    # ── Step 2: EXIF ─────────────────────────────────────────────
    metadata_score, exif_indicators, _software = analyze_exif(image_bytes)
    indicators.extend(exif_indicators)

    # ── Step 3: CNN Inference (photo head) ───────────────────────
    cnn_probs = _run_cnn_inference(ela_image)

    # ── Step 4: TruFor (text / content manipulation) ─────────────
    trufor_prob, trufor_downscaled = _run_trufor(image_bytes)

    # ── Step 5: Score Computation (deterministic) ────────────────

    # photo_replacement: 0-65
    photo_raw = cnn_probs.get("photo_replacement", 0.0)
    photo_score = max(0, min(PHOTO_MAX_POINTS, int(round(photo_raw * PHOTO_MAX_POINTS))))
    if photo_raw > 0.5:
        indicators.append(
            f"Tampering detected by AI model (confidence: {photo_raw:.0%})"
        )

    # text_manipulation: 0-65 (TruFor). 0 if TruFor unavailable — see details["trufor_available"].
    text_score = 0
    if trufor_prob is not None:
        text_score = max(0, min(TEXT_MAX_POINTS, int(round(trufor_prob * TEXT_MAX_POINTS))))
        if trufor_prob > 0.5:
            indicators.append(
                f"Text/content manipulation detected (TruFor confidence: {trufor_prob:.0%})"
            )

    # metadata_anomaly: 0-10 (already computed from EXIF)
    metadata_score = max(0, min(METADATA_MAX_POINTS, metadata_score))

    # compression_anomaly: 0-15 (ELA heuristic)
    # High ELA max_diff on a JPEG suggests re-compression at different quality
    if ela_max_diff > 200:
        comp_ela_score = 15
        indicators.append("Significant compression level variation detected across image regions")
    elif ela_max_diff > 100:
        comp_ela_score = 10
    elif ela_max_diff > 50:
        comp_ela_score = 5
    else:
        comp_ela_score = 0
    compression_score = max(0, min(COMPRESSION_MAX_POINTS, comp_ela_score))

    # ── Step 6: Total Score ──────────────────────────────────────
    total_score = photo_score + text_score + metadata_score + compression_score
    total_score = max(0, min(TOTAL_CAP, total_score))

    # ── Step 7: Suspicious Flag ──────────────────────────────────
    suspicious = total_score >= SUSPICIOUS_THRESHOLD

    # ── Step 8: Visualization (optional, best-effort) ────────────
    visualization_path = None
    if total_score >= 30:
        # Only generate visualization if there's something worth showing
        visualization_path = generate_ela_visualization(ela_image)

    ela_image.close()

    # ── Step 9: Clean indicators if no strong anomaly ────────────
    if total_score < 15:
        indicators = []

    return {
        "score": total_score,
        "suspicious": suspicious,
        "details": {
            "photo_replacement": photo_score,
            "text_manipulation": text_score,
            "metadata_anomaly": metadata_score,
            "compression_anomaly": compression_score,
            "trufor_probability": None if trufor_prob is None else round(trufor_prob, 4),
            "trufor_available": trufor_prob is not None,
            "trufor_downscaled": trufor_downscaled,
            "visualization": visualization_path,
        },
        "indicators": indicators,
    }