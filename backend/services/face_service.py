"""
Face verification service using lightweight OpenCV (no ML model downloads).

Strategy:
  1. Detect faces in both images using Haar cascades (built into opencv-headless).
  2. Extract ORB keypoint descriptors from each detected face crop.
  3. Match descriptors with BFMatcher and derive a similarity score.

Memory footprint: ~5 MB (Haar XML bundled with OpenCV). No PyTorch / TF needed.
"""
from typing import Dict, Any, Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV ships the cascade XML; no download needed.
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_face(image_path: str) -> Optional[np.ndarray]:
    """Return a grayscale face crop (resized to 100×100) or None."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    crop = gray[y : y + h, x : x + w]
    return cv2.resize(crop, (100, 100))


def _orb_similarity(face1: np.ndarray, face2: np.ndarray) -> float:
    """
    Compare two face crops using ORB descriptors + BFMatcher.
    Returns a similarity score in [0.0, 1.0].
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(face1, None)
    kp2, des2 = orb.detectAndCompute(face2, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0

    # Good matches: distance < 60 out of max 256
    good = [m for m in matches if m.distance < 60]
    ratio = len(good) / max(len(matches), 1)
    # Scale: >0.35 good match, <0.10 poor match
    similarity = min(1.0, ratio / 0.35)
    return round(similarity, 2)


def verify_faces(
    document_image_path: str, live_face_image_path: Optional[str]
) -> Dict[str, Any]:
    """Verify that the face on the document matches the traveller selfie."""
    if not live_face_image_path:
        return {
            "available": False,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "message": "Traveller face image not provided",
        }

    face_doc = _detect_face(document_image_path)
    face_live = _detect_face(live_face_image_path)

    doc_detected = face_doc is not None
    live_detected = face_live is not None

    if not doc_detected or not live_detected:
        return {
            "available": True,
            "face_detected_document": doc_detected,
            "face_detected_live": live_detected,
            "match": False,
            "similarity": 0.0,
            "message": "Face not detected in one or both images",
        }

    similarity = _orb_similarity(face_doc, face_live)
    # Threshold chosen to balance FP/FN on passport-style photos
    is_match = similarity >= 0.45

    return {
        "available": True,
        "face_detected_document": True,
        "face_detected_live": True,
        "match": is_match,
        "similarity": similarity,
        "message": "Faces appear to match" if is_match else "Faces do not match sufficiently",
    }

