"""
Risk Scoring Engine.
"""
from typing import Dict, Any

def calculate_risk(
    ocr_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    tampering_result: Dict[str, Any],
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    score = 0.0
    reasons = []

    # 1. Validation Failures
    if not validation_result.get("valid", True):
        # Major validation failure
        points = min(25.0, 100 - validation_result.get("score", 100))
        if points > 0:
            score += points
            reasons.append({
                "factor": "validation_failure",
                "points": points,
                "message": "Document validation checks failed"
            })
    
    # Check for expired document explicitly in checks
    checks = validation_result.get("checks", [])
    is_expired = False
    for check in checks:
        if check.get("name") == "Expiry date" and check.get("status") == "fail":
            is_expired = True
    
    if is_expired:
        score += 30.0
        reasons.append({
            "factor": "document_expired",
            "points": 30.0,
            "message": "Document is expired"
        })

    # 2. OCR Confidence
    ocr_conf = ocr_result.get("confidence", 1.0)
    if ocr_conf < 0.6:
        score += 10.0
        reasons.append({
            "factor": "low_ocr_confidence",
            "points": 10.0,
            "message": "OCR text extraction confidence is very low"
        })

    # 3. Tampering
    tamp_score = tampering_result.get("score", 0.0)
    tamp_points = tamp_score * 0.45
    if tamp_points > 0:
        score += tamp_points
        reasons.append({
            "factor": "tampering_detected",
            "points": round(tamp_points, 2),
            "message": f"Document shows signs of tampering (score: {tamp_score})"
        })

    # 4. Face Verification
    if face_result.get("available"):
        if face_result.get("match") is False:
            score += 40.0
            reasons.append({
                "factor": "face_mismatch",
                "points": 40.0,
                "message": "Traveller face does not sufficiently match the document portrait"
            })
        elif face_result.get("face_detected_document") is False or face_result.get("face_detected_live") is False:
             # Face missing in one of them when it was supposed to be there
             score += 20.0
             reasons.append({
                "factor": "face_not_detected",
                "points": 20.0,
                "message": "Face could not be detected in one or both images"
            })

    # Cap at 100
    score = min(100.0, round(score, 2))

    # Risk bands:
    # 0-30 LOW (CLEAR)
    # 31-60 REVIEW (MANUAL_REVIEW)
    # 61-100 HIGH (ESCALATE)
    
    if score <= 30:
        level = "LOW"
        decision = "CLEAR"
    elif score <= 60:
        level = "REVIEW"
        decision = "MANUAL_REVIEW"
    else:
        level = "HIGH"
        decision = "ESCALATE"

    return {
        "score": score,
        "level": level,
        "decision": decision,
        "reasons": reasons
    }
