"""
Image Quality Assessment Service.

Evaluates document images for issues that could degrade OCR and face verification.
"""
import logging
from typing import Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def analyze_image_quality(image_path: str) -> Dict[str, Any]:
    """
    Evaluate image quality and return scores and flags.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "score": 0,
            "acceptable": False,
            "blur_score": 0,
            "glare_score": 0,
            "brightness_score": 0,
            "resolution_ok": False,
            "document_visible": False,
            "issues": ["Could not load image"]
        }
        
    issues = []
    
    # 1. Resolution
    h, w = img.shape[:2]
    resolution_ok = h >= 300 and w >= 400
    if not resolution_ok:
        issues.append(f"Image resolution too low ({w}x{h})")
        
    # 2. Blur detection (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = blur_score > 50.0  # arbitrary threshold
    if not blur_ok:
        issues.append("Image appears significantly blurred")
        
    # 3. Brightness & Over/Under exposure
    brightness_score = np.mean(gray)
    if brightness_score < 40:
        issues.append("Image is severely underexposed (too dark)")
    elif brightness_score > 220:
        issues.append("Image is severely overexposed (too bright)")
        
    # 4. Glare detection (high intensity pixels)
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = cv2.countNonZero(glare_mask) / (h * w)
    glare_score = glare_ratio * 100
    if glare_score > 5.0: # more than 5% of image is pure white glare
        issues.append("Significant glare/reflections detected")
        
    # 5. Document framing (Edge density)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / (h * w)
    document_visible = edge_density > 0.02
    if not document_visible:
        issues.append("Document structure barely visible")
        
    # Calculate overall score 0-100
    score = 100
    if not resolution_ok: score -= 20
    if not blur_ok: score -= 30
    if brightness_score < 40 or brightness_score > 220: score -= 20
    if glare_score > 5.0: score -= 15
    if not document_visible: score -= 30
    
    score = max(0, score)
    acceptable = score >= 50
    
    if not acceptable and not issues:
        issues.append("Image quality too low for reliable screening")
        
    return {
        "score": score,
        "acceptable": acceptable,
        "blur_score": round(blur_score, 2),
        "glare_score": round(glare_score, 2),
        "brightness_score": round(brightness_score, 2),
        "resolution_ok": resolution_ok,
        "document_visible": document_visible,
        "issues": issues
    }
