"""
Face-aware tamper detection service.

Key guarantees:
- Never raises from analyze_tampering().
- Keeps TruFor probability directly as text/content anomaly signal.
- Adds explicit face-gated photo replacement analysis.
- Supports legacy checkpoints (tamper_multihead_resnet18.pth / tamper_resnet18.pth).
- Uses bounded calibrated fusion (no additive overflow).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from torchvision import models, transforms

try:
    import exifread

    HAS_EXIFREAD = True
except Exception:
    HAS_EXIFREAD = False

try:
    import cv2

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# =============================================================================
# Configuration
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))

SUSPICIOUS_THRESHOLD = 55
TOTAL_CAP = 100

DEFAULT_INFERENCE_META = {
    "model_name": "compact_tamper_net_v1",
    "input_size": 224,
    "ela_quality": 90,
    "ela_scale": 20,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "photo_face_aggregate": "max",
    "fusion_weights": {
        "photo": 0.42,
        "document": 0.30,
        "trufor": 0.24,
        "metadata": 0.04,
    },
    "temperature": {
        "photo": 1.0,
        "document": 1.0,
    },
    "prob_to_percent": "round(clamp(p,0,1)*100)",
    "overall_formula": "1 - Π_i(1 - clamp(w_i*p_i, 0, 1))",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


TRUFOR_MAX_SIDE = _env_int("TRUFOR_MAX_SIDE", 0)
_TRUFOR_OOM_RETRY_SIDE = 2048


# =============================================================================
# Helpers
# =============================================================================


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _to_pct(v: float) -> int:
    return int(round(_clamp01(v) * 100.0))


def _safe_softmax_tamper(logits: torch.Tensor) -> float:
    try:
        return float(torch.softmax(logits, dim=1)[0, 1].item())
    except Exception:
        return 0.0


def _safe_sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + np.exp(-x))


def _resolve_device() -> torch.device:
    name = os.environ.get("TAMPER_DEVICE", "cpu")
    try:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev
    except Exception:
        return torch.device("cpu")


def _safe_open_rgb(image_bytes: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.load()
        return img
    except Exception:
        return None


def _crop_with_margin(img: Image.Image, box: Tuple[int, int, int, int], margin_ratio: float = 0.2) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)
    if nx2 <= nx1 or ny2 <= ny1:
        return img.copy()
    return img.crop((nx1, ny1, nx2, ny2))


# =============================================================================
# Model definitions and loading
# =============================================================================


class CompactTamperNet(nn.Module):
    """
    Compact dual-head network used for new checkpoints.

    Input is 6 channels:
    - RGB (3)
    - ELA (3)
    Outputs:
    - photo_replacement logits (2)
    - document_tamper logits (2)
    """

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1_w = base.conv1.weight.data.clone()
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = conv1_w
            base.conv1.weight[:, 3:] = conv1_w

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512
        self.photo_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )
        self.doc_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x6: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x6).flatten(1)
        return {
            "photo_replacement": self.photo_head(feat),
            "document_tamper": self.doc_head(feat),
        }


class LegacyMultiHeadResNet18(nn.Module):
    """Compatibility model for historical multihead checkpoint."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        feat_dim = 512
        self.head_photo = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.head_text = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.head_comp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        f = self.features(x).flatten(1)
        return {
            "photo_replacement": self.head_photo(f),
            "text_manipulation": self.head_text(f),
            "compression_anomaly": self.head_comp(f),
        }


_device = _resolve_device()
_model = None
_model_loaded = False
_model_type = "none"
_model_lock = threading.Lock()
_model_meta = dict(DEFAULT_INFERENCE_META)


def _load_meta() -> Dict:
    meta_path = os.path.join(_HERE, "inference_meta.json")
    meta = dict(DEFAULT_INFERENCE_META)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                disk = json.load(f)
            if isinstance(disk, dict):
                meta.update(disk)
                meta.setdefault("fusion_weights", DEFAULT_INFERENCE_META["fusion_weights"])
                meta.setdefault("temperature", DEFAULT_INFERENCE_META["temperature"])
        except Exception:
            pass
    return meta


_model_meta = _load_meta()


def _build_rgb_transform() -> transforms.Compose:
    size = int(_model_meta.get("input_size", 224))
    mean = _model_meta.get("normalize_mean", [0.485, 0.456, 0.406])
    std = _model_meta.get("normalize_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


_transform_rgb = _build_rgb_transform()


def _safe_load_torch(path: str):
    try:
        return torch.load(path, map_location=_device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=_device)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else None


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _checkpoint_candidates() -> List[Tuple[str, str]]:
    dirs = [
        _HERE,
        os.path.join(_HERE, "cloud", "checkpoints"),
        "/app/services/tamper",
        "/app/services/tamper/cloud/checkpoints",
    ]
    names = [
        "tamper_compact_multisignal.pth",
        "tamper_multihead_resnet18.pth",
        "tamper_resnet18.pth",
    ]
    out = []
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                out.append((n, p))
    return out


def _load_model() -> bool:
    global _model, _model_loaded, _model_type

    for ckpt_name, ckpt_path in _checkpoint_candidates():
        try:
            raw = _safe_load_torch(ckpt_path)
            state = _extract_state_dict(raw)
            if not isinstance(state, dict):
                continue
            state = _strip_module_prefix(state)

            # New compact model
            if "photo_head.1.weight" in state and "doc_head.1.weight" in state:
                m = CompactTamperNet(pretrained=False)
                m.load_state_dict(state, strict=True)
                m.to(_device).eval()
                _model = m
                _model_loaded = True
                _model_type = "compact_multisignal"
                return True

            # Legacy multihead
            if "head_photo.1.weight" in state and "head_comp.1.weight" in state:
                m = LegacyMultiHeadResNet18()
                m.load_state_dict(state, strict=True)
                m.to(_device).eval()
                _model = m
                _model_loaded = True
                _model_type = "legacy_multihead"
                return True

            # Legacy single-head
            if "fc.weight" in state:
                m = models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 2)
                m.load_state_dict(state, strict=True)
                m.to(_device).eval()
                _model = m
                _model_loaded = True
                _model_type = "legacy_singlehead"
                return True

        except Exception:
            continue

    _model = None
    _model_loaded = False
    _model_type = "none"
    return False


_load_model()


# =============================================================================
# Face detector
# =============================================================================

_face_detector = None
_face_detector_available = False


def _init_face_detector() -> None:
    global _face_detector, _face_detector_available
    if not HAS_CV2:
        return
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.isfile(cascade_path):
            return
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return
        _face_detector = detector
        _face_detector_available = True
    except Exception:
        _face_detector = None
        _face_detector_available = False


_init_face_detector()


def _detect_faces(img: Image.Image) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """
    Returns list of face boxes in (x1,y1,x2,y2), and detector status string.
    """
    if not _face_detector_available:
        return [], "unavailable"

    try:
        arr = np.asarray(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        min_side = max(24, int(0.06 * min(w, h)))
        faces = _face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        boxes = []
        for (x, y, bw, bh) in faces:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes, "ok"
    except Exception:
        return [], "error"


# =============================================================================
# TruFor integration
# =============================================================================

_trufor_model = None
_trufor_device = _device
_trufor_lock = threading.Lock()


def _find_trufor_paths() -> Optional[Tuple[str, str]]:
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
def _trufor_import_context(src_dir: str):
    top_names = set()
    for entry in os.listdir(src_dir):
        full = os.path.join(src_dir, entry)
        if os.path.isdir(full):
            top_names.add(entry)
        elif entry.endswith(".py"):
            top_names.add(entry[:-3])

    def _is_trufor_name(mod_name: str) -> bool:
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


def _load_trufor() -> bool:
    global _trufor_model
    try:
        import timm  # noqa: F401
        import yacs  # noqa: F401
    except Exception:
        return False

    found = _find_trufor_paths()
    if found is None:
        return False
    src_dir, weights_path = found

    try:
        with _trufor_import_context(src_dir):
            from config import _C as trufor_cfg
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

            trufor_cfg.defrost()
            trufor_cfg.merge_from_file(os.path.join(src_dir, "trufor.yaml"))
            if hasattr(trufor_cfg.MODEL, "PRETRAINED"):
                trufor_cfg.MODEL.PRETRAINED = ""
            trufor_cfg.freeze()

            model = confcmx(cfg=trufor_cfg)
            try:
                ckpt = torch.load(weights_path, map_location=_trufor_device, weights_only=True)
            except Exception:
                ckpt = torch.load(weights_path, map_location=_trufor_device, weights_only=False)

            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state)
            model.to(_trufor_device).eval()

            with torch.no_grad():
                model(torch.zeros(1, 3, 512, 512, device=_trufor_device))

            _trufor_model = model
            return True
    except Exception:
        _trufor_model = None
        return False


_load_trufor()


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def _trufor_forward(img_bytes: bytes, max_side: int) -> float:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    rgb = torch.from_numpy(arr).float().div_(256.0).unsqueeze(0).to(_trufor_device)

    found = _find_trufor_paths()
    if not found:
        raise RuntimeError("TruFor paths unavailable")

    with _trufor_import_context(found[0]):
        with torch.no_grad():
            out = _trufor_model(rgb)

    det = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
    if det is None:
        raise RuntimeError("TruFor output missing detection score")

    return float(torch.sigmoid(det).flatten()[0].item())


def _run_trufor(img_bytes: bytes) -> Tuple[Optional[float], bool]:
    if _trufor_model is None:
        return None, False

    with _trufor_lock:
        try:
            return _trufor_forward(img_bytes, TRUFOR_MAX_SIDE), bool(TRUFOR_MAX_SIDE)
        except Exception as e:
            if _is_oom(e) and not TRUFOR_MAX_SIDE:
                if _trufor_device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    return _trufor_forward(img_bytes, _TRUFOR_OOM_RETRY_SIDE), True
                except Exception:
                    return None, False
            return None, False


# =============================================================================
# Image signals
# =============================================================================


def compute_ela(img_rgb: Image.Image, quality: int = 90, scale: int = 20, size: int = 224) -> Tuple[Image.Image, int]:
    """Deterministic ELA; never raises."""
    try:
        buf = io.BytesIO()
        img_rgb.save(buf, "JPEG", quality=int(quality))
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")

        diff = np.abs(np.asarray(img_rgb, np.int16) - np.asarray(compressed, np.int16))
        max_diff = int(diff.max())
        ela_arr = np.clip(diff * int(scale), 0, 255).astype(np.uint8)
        ela_img = Image.fromarray(ela_arr).resize((int(size), int(size)), Image.BILINEAR)
        return ela_img, max_diff
    except Exception:
        return Image.new("RGB", (int(size), int(size)), (0, 0, 0)), 0


def compute_residual_map(img_rgb: Image.Image, size: int = 224) -> Image.Image:
    """Lightweight high-pass residual map used for document head robustness."""
    try:
        g = ImageOps.grayscale(img_rgb.resize((size, size), Image.BILINEAR))
        blur = g.filter(ImageFilter.GaussianBlur(radius=1.2))
        g_np = np.asarray(g, dtype=np.float32)
        b_np = np.asarray(blur, dtype=np.float32)
        resid = np.clip(np.abs(g_np - b_np) * 4.0, 0, 255).astype(np.uint8)
        return Image.merge("RGB", (Image.fromarray(resid),) * 3)
    except Exception:
        return Image.new("RGB", (size, size), (0, 0, 0))


def analyze_exif(img_bytes: bytes) -> Tuple[float, List[str], Optional[str]]:
    """
    Weak metadata signal only (0..1). EXIF absence is intentionally low-impact.
    """
    if not HAS_EXIFREAD:
        return 0.0, [], None

    try:
        tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
    except Exception:
        return 0.0, [], None

    score = 0.0
    indicators: List[str] = []
    software = None

    software_tag = tags.get("Image Software")
    if software_tag:
        software = str(software_tag)
        suspicious_editors = ["photoshop", "gimp", "affinity", "pixlr", "canva", "snapseed"]
        if any(ed in software.lower() for ed in suspicious_editors):
            score += 0.45
            indicators.append(f"Metadata references editing software ({software})")

    dt_orig = tags.get("EXIF DateTimeOriginal")
    dt_mod = tags.get("Image DateTime")
    if dt_orig and dt_mod and str(dt_orig) != str(dt_mod):
        score += 0.25
        indicators.append("Metadata timestamps differ between original and current image")

    # EXIF absence is intentionally weak
    if not tags:
        score += 0.10

    return _clamp01(score), indicators, software


def _temperature_scale(prob: float, t: float) -> float:
    t = max(0.1, float(t))
    logit = np.log(max(1e-6, prob) / max(1e-6, 1.0 - prob))
    return _safe_sigmoid(logit / t)


def _noisy_or(weighted_probs: Dict[str, float]) -> float:
    v = 1.0
    for p in weighted_probs.values():
        v *= 1.0 - _clamp01(p)
    return _clamp01(1.0 - v)


def _run_model(rgb_img: Image.Image, ela_img: Image.Image) -> Dict[str, float]:
    if _model is None:
        return {"photo_replacement": 0.0, "document_tamper": 0.0}

    with _model_lock:
        try:
            rgb_t = _transform_rgb(rgb_img).unsqueeze(0).to(_device)

            if _model_type == "compact_multisignal":
                ela_t = _transform_rgb(ela_img).unsqueeze(0).to(_device)
                x6 = torch.cat([rgb_t, ela_t], dim=1)
                with torch.no_grad():
                    logits = _model(x6)
                p_photo = _safe_softmax_tamper(logits["photo_replacement"])
                p_doc = _safe_softmax_tamper(logits["document_tamper"])
                return {"photo_replacement": p_photo, "document_tamper": p_doc}

            if _model_type == "legacy_multihead":
                with torch.no_grad():
                    logits = _model(_transform_rgb(ela_img).unsqueeze(0).to(_device))
                p_photo = _safe_softmax_tamper(logits["photo_replacement"])
                p_doc = _safe_softmax_tamper(logits.get("compression_anomaly", logits["photo_replacement"]))
                return {"photo_replacement": p_photo, "document_tamper": p_doc}

            if _model_type == "legacy_singlehead":
                with torch.no_grad():
                    logits = _model(_transform_rgb(ela_img).unsqueeze(0).to(_device))
                p_photo = _safe_softmax_tamper(logits)
                return {"photo_replacement": p_photo, "document_tamper": p_photo * 0.8}
        except Exception:
            return {"photo_replacement": 0.0, "document_tamper": 0.0}

    return {"photo_replacement": 0.0, "document_tamper": 0.0}


def _aggregate_face_probs(probs: List[float]) -> float:
    if not probs:
        return 0.0
    agg = _model_meta.get("photo_face_aggregate", "max")
    if agg == "mean":
        return _clamp01(float(np.mean(probs)))
    if agg == "p90":
        return _clamp01(float(np.percentile(probs, 90)))
    return _clamp01(float(np.max(probs)))


def _empty_result(indicators: Optional[List[str]] = None) -> Dict:
    return {
        "score": 0,
        "suspicious": False,
        "details": {
            "photo_replacement": 0,
            "document_tamper": 0,
            "text_manipulation": 0,
            "metadata_anomaly": 0,
            "compression_anomaly": 0,
            "photo_probability": 0,
            "document_probability": 0,
            "trufor_probability": 0,
            "face_count": 0,
            "no_face_detected": True,
            "photo_available": False,
            "trufor_available": False,
            "trufor_downscaled": False,
            "model_status": get_model_status(),
            "face_detector_status": "unavailable" if not _face_detector_available else "ok",
            "visualization": None,
        },
        "indicators": list(indicators or []),
    }


def get_model_status() -> Dict[str, object]:
    return {
        "tamper_model_loaded": bool(_model is not None),
        "tamper_model_type": _model_type,
        "trufor_loaded": bool(_trufor_model is not None),
        "face_detector_available": bool(_face_detector_available),
        "device": str(_device),
        "meta_version": _model_meta.get("model_name", "unknown"),
    }


def generate_ela_visualization(ela_image: Image.Image) -> Optional[str]:
    try:
        vis_filename = f"ela_{uuid.uuid4().hex[:8]}.png"
        vis_path = os.path.join(tempfile.gettempdir(), vis_filename)
        ela_image.save(vis_path, "PNG")
        return vis_path
    except Exception:
        return None


# =============================================================================
# Public API
# =============================================================================


def analyze_tampering(image_bytes: bytes) -> Dict:
    """Never raises."""
    try:
        return _analyze_tampering_impl(image_bytes)
    except Exception:
        traceback.print_exc()
        return _empty_result(["Internal tamper analysis failure; returned safe default"])


def _analyze_tampering_impl(image_bytes: bytes) -> Dict:
    if not image_bytes or len(image_bytes) < 32:
        return _empty_result(["Image too small or empty to analyze"])

    rgb = _safe_open_rgb(image_bytes)
    if rgb is None:
        return _empty_result(["Invalid or unreadable image input"])

    indicators: List[str] = []

    meta_size = int(_model_meta.get("input_size", 224))
    ela_quality = int(_model_meta.get("ela_quality", 90))
    ela_scale = int(_model_meta.get("ela_scale", 20))

    ela_full, ela_max_diff = compute_ela(rgb, quality=ela_quality, scale=ela_scale, size=meta_size)
    residual_full = compute_residual_map(rgb, size=meta_size)

    metadata_prob, exif_indicators, _ = analyze_exif(image_bytes)
    indicators.extend(exif_indicators)

    face_boxes, face_detector_status = _detect_faces(rgb)
    face_count = len(face_boxes)
    no_face_detected = face_count == 0

    # Document/global model path (runs irrespective of face detection)
    doc_signals = _run_model(rgb.resize((meta_size, meta_size), Image.BILINEAR), ela_full)
    doc_prob = _clamp01(doc_signals.get("document_tamper", 0.0))

    # Lightweight residual uplift to help non-face region tamper sensitivity
    residual_prob = _clamp01(float(np.asarray(residual_full).mean() / 255.0))
    doc_prob = _clamp01(0.85 * doc_prob + 0.15 * residual_prob)

    # Photo/face path
    per_face_probs: List[float] = []
    if not no_face_detected:
        for box in face_boxes:
            crop_rgb = _crop_with_margin(rgb, box, margin_ratio=0.2)
            crop_ela, _ = compute_ela(crop_rgb, quality=ela_quality, scale=ela_scale, size=meta_size)
            face_signals = _run_model(crop_rgb.resize((meta_size, meta_size), Image.BILINEAR), crop_ela)
            per_face_probs.append(_clamp01(face_signals.get("photo_replacement", 0.0)))

    photo_available = not no_face_detected
    photo_prob = _aggregate_face_probs(per_face_probs) if photo_available else 0.0

    # TruFor path must stay direct for text/content anomaly
    trufor_prob_raw, trufor_downscaled = _run_trufor(image_bytes)
    trufor_prob = _clamp01(trufor_prob_raw) if trufor_prob_raw is not None else 0.0
    trufor_available = trufor_prob_raw is not None

    # Temperature calibration
    temp = _model_meta.get("temperature", {}) or {}
    photo_cal = _temperature_scale(photo_prob, temp.get("photo", 1.0)) if photo_available else 0.0
    doc_cal = _temperature_scale(doc_prob, temp.get("document", 1.0))

    # Deterministic integer percentages
    photo_pct = _to_pct(photo_cal) if photo_available else 0
    doc_pct = _to_pct(doc_cal)
    trufor_pct = _to_pct(trufor_prob) if trufor_available else 0
    metadata_pct = _to_pct(metadata_prob)

    # Keep legacy compression key; still deterministic and bounded
    if ela_max_diff > 200:
        compression_pct = 85
    elif ela_max_diff > 100:
        compression_pct = 60
    elif ela_max_diff > 50:
        compression_pct = 30
    else:
        compression_pct = 0

    # Weighted noisy-OR fusion avoids overflow and score inflation
    fw = _model_meta.get("fusion_weights", DEFAULT_INFERENCE_META["fusion_weights"])
    weighted_inputs = {
        "photo": fw.get("photo", 0.42) * (photo_cal if photo_available else 0.0),
        "document": fw.get("document", 0.30) * doc_cal,
        "trufor": fw.get("trufor", 0.24) * (trufor_prob if trufor_available else 0.0),
        "metadata": fw.get("metadata", 0.04) * metadata_prob,
    }
    overall_prob = _noisy_or(weighted_inputs)
    total_score = _to_pct(overall_prob)
    suspicious = total_score >= SUSPICIOUS_THRESHOLD

    if no_face_detected:
        indicators.append("No face detected; photo replacement analysis was skipped")
    elif photo_cal > 0.5:
        indicators.append(f"Face/photo replacement risk detected (confidence {photo_pct}%)")

    if doc_cal > 0.5:
        indicators.append(f"Document-level tamper risk detected (confidence {doc_pct}%)")

    if trufor_available and trufor_prob > 0.5:
        indicators.append(f"Text/content anomaly detected by TruFor (confidence {trufor_pct}%)")

    visualization_path = generate_ela_visualization(ela_full) if total_score >= 30 else None

    # Clean low-signal noise
    if total_score < 12:
        indicators = []

    return {
        "score": max(0, min(TOTAL_CAP, total_score)),
        "suspicious": suspicious,
        "details": {
            "photo_replacement": photo_pct,
            "document_tamper": doc_pct,
            "text_manipulation": trufor_pct,
            "metadata_anomaly": metadata_pct,
            "compression_anomaly": int(compression_pct),
            "photo_probability": photo_pct,
            "document_probability": doc_pct,
            "trufor_probability": trufor_pct,
            "face_count": face_count,
            "no_face_detected": no_face_detected,
            "photo_available": photo_available,
            "trufor_available": trufor_available,
            "trufor_downscaled": trufor_downscaled,
            "model_status": get_model_status(),
            "face_detector_status": face_detector_status,
            "visualization": visualization_path,
        },
        "indicators": indicators,
    }
