"""
Face verification service using OpenCV YuNet (detection) and SFace (recognition).

Strategy:
  1. Download lightweight ONNX models (YuNet and SFace) on first run if missing.
  2. Detect faces using YuNet (much stronger than Haar cascades).
  3. Extract 128D face embeddings using SFace.
  4. Compare embeddings using cosine similarity.

Memory footprint: ~30 MB. No heavy ML frameworks needed. Very stable for Render free tier.
"""
import os
import urllib.request
import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_detector = None
_recognizer = None

def _download_model(url: str, path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Downloading model {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        logger.info(f"Downloaded {os.path.basename(path)}")

def _get_models():
    global _detector, _recognizer
    if _detector is None or _recognizer is None:
        _download_model(YUNET_URL, YUNET_PATH)
        _download_model(SFACE_URL, SFACE_PATH)
        
        # Initialize YuNet
        _detector = cv2.FaceDetectorYN_create(
            YUNET_PATH,
            "",
            (320, 320),
            0.8,
            0.3,
            5000
        )
        # Initialize SFace
        _recognizer = cv2.FaceRecognizerSF_create(SFACE_PATH, "")
    return _detector, _recognizer

def _get_face_embedding(image_path: str) -> Optional[Tuple[np.ndarray, float]]:
    """Detects face and returns (embedding, quality_score)."""
    detector, recognizer = _get_models()
    
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w, _ = img.shape
    detector.setInputSize((w, h))
    
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
        
    # Get the face with highest confidence (faces format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score])
    face = max(faces, key=lambda f: f[-1])
    confidence = float(face[-1])
    
    # Align crop and extract features
    aligned_face = recognizer.alignCrop(img, face)
    embedding = recognizer.feature(aligned_face)
    
    return embedding, confidence

def verify_faces(
    document_image_path: str, live_face_image_path: Optional[str]
) -> Dict[str, Any]:
    """Verify that the face on the document matches the traveller selfie."""
    if not live_face_image_path:
        return {
            "available": True,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "quality_document": None,
            "quality_live": None,
            "message": "Traveller face image not provided",
        }

    try:
        doc_result = _get_face_embedding(document_image_path)
        live_result = _get_face_embedding(live_face_image_path)
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return {
            "available": False,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "quality_document": None,
            "quality_live": None,
            "message": "Face verification failed internally",
        }

    doc_detected = doc_result is not None
    live_detected = live_result is not None

    if not doc_detected or not live_detected:
        return {
            "available": True,
            "face_detected_document": doc_detected,
            "face_detected_live": live_detected,
            "match": False,
            "similarity": 0.0,
            "quality_document": doc_result[1] if doc_detected else 0.0,
            "quality_live": live_result[1] if live_detected else 0.0,
            "message": "Face not detected in one or both images",
        }

    doc_embedding, doc_quality = doc_result
    live_embedding, live_quality = live_result

    # SFace returns L2 normalized embeddings, so cosine similarity is just dot product
    # Or use recognizer.match
    recognizer = _get_models()[1]
    # match type: 0 for cosine, 1 for L2
    similarity = recognizer.match(doc_embedding, live_embedding, cv2.FaceRecognizerSF_FR_COSINE)
    
    # SFace cosine similarity threshold is typically around 0.363 for true positive rate
    is_match = similarity >= 0.363
    
    # Scale similarity to look more intuitive (0.363 -> ~0.7, 1.0 -> 1.0)
    # This is optional but helps with user interpretation.
    scaled_sim = min(1.0, max(0.0, (similarity + 0.5) / 1.5)) if similarity > 0 else 0.0

    return {
        "available": True,
        "face_detected_document": True,
        "face_detected_live": True,
        "match": is_match,
        "similarity": float(scaled_sim),
        "quality_document": float(doc_quality),
        "quality_live": float(live_quality),
        "message": "Faces appear to match" if is_match else "Face mismatch detected",
    }
