"""
Template & Layout Consistency Service.

Validates that the document structure follows expected geometric patterns.
"""
import logging
from typing import Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def analyze_document_layout(image_path: str, document_type: str) -> Dict[str, Any]:
    """
    Check if the document layout is consistent with its type.
    Focuses mainly on generic passport/ID properties (e.g. portrait position, MRZ position).
    """
    if document_type not in ("passport", "national_id"):
        return {
            "available": False,
            "score": 0,
            "layout_consistent": None,
            "checks": []
        }
        
    img = cv2.imread(image_path)
    if img is None:
        return {
            "available": False,
            "score": 0,
            "layout_consistent": False,
            "checks": [{"name": "Image load", "status": "fail", "message": "Could not read image"}]
        }
        
    h, w = img.shape[:2]
    checks = []
    
    # 1. Aspect Ratio check
    # ID cards are typically ~1.58 (CR80), Passports are ~1.4. Let's accept 1.2 to 1.8.
    # Allowing orientation either way (w/h or h/w)
    ratio = max(w/h, h/w)
    if 1.2 <= ratio <= 1.8:
        checks.append({"name": "Aspect Ratio", "status": "pass", "message": f"Aspect ratio ({ratio:.2f}) is plausible for {document_type}"})
    else:
        checks.append({"name": "Aspect Ratio", "status": "warn", "message": f"Aspect ratio ({ratio:.2f}) is unusual for standard documents"})
        
    # 2. Text region distribution (MRZ detection area)
    # The MRZ is expected at the bottom. We can check if there's significant edge density in the bottom 20%.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Assuming landscape mode, MRZ is at bottom. If portrait, we don't know easily without rotation.
    if w > h: # Landscape
        bottom_region = edges[int(h*0.8):, :]
        mrz_edge_density = cv2.countNonZero(bottom_region) / (bottom_region.shape[0] * bottom_region.shape[1])
        if mrz_edge_density > 0.05:
            checks.append({"name": "MRZ location", "status": "pass", "message": "Text/Structure detected in expected MRZ lower region"})
        else:
            checks.append({"name": "MRZ location", "status": "warn", "message": "Expected structure missing in lower MRZ region"})
            
    # Compute score
    passed = sum(1 for c in checks if c["status"] == "pass")
    score = (passed / max(1, len(checks))) * 100
    
    return {
        "available": True,
        "score": int(score),
        "layout_consistent": score >= 50,
        "checks": checks
    }
