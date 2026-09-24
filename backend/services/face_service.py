"""
Face verification service using DeepFace.
"""
from typing import Dict, Any, Optional
import logging

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not installed or failed to import. Verification will fall back to error or mock logic.")

def verify_faces(document_image_path: str, live_face_image_path: Optional[str]) -> Dict[str, Any]:
    if not live_face_image_path:
        return {
            "available": False,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "message": "Traveller face image not provided"
        }
    
    if not DEEPFACE_AVAILABLE:
        return {
            "available": False,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "message": "Face recognition library unavailable"
        }

    try:
        # We use a permissive distance metric. DeepFace verify returns a dictionary.
        result = DeepFace.verify(
            img1_path=document_image_path,
            img2_path=live_face_image_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True
        )

        # In DeepFace, distance is smaller for closer matches. 
        # Let's mock a similarity score between 0 and 1. 1 - distance.
        distance = result.get("distance", 1.0)
        similarity = max(0.0, 1.0 - distance)
        
        is_match = bool(result.get("verified", False))
        
        return {
            "available": True,
            "face_detected_document": True,  # If it didn't throw an error, faces were detected
            "face_detected_live": True,
            "match": is_match,
            "similarity": round(similarity, 2),
            "message": "Faces appear to match" if is_match else "Faces do not match sufficiently"
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "face could not be detected" in error_msg:
            return {
                "available": True,
                "face_detected_document": False,
                "face_detected_live": False,
                "match": False,
                "similarity": 0.0,
                "message": "Face not detected in one or both images"
            }
        
        logging.error(f"Face verification error: {e}")
        return {
            "available": False,
            "face_detected_document": None,
            "face_detected_live": None,
            "match": None,
            "similarity": None,
            "message": "Internal face comparison error"
        }
