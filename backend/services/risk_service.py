"""
Risk Scoring Engine.
"""
from typing import Dict, Any

def calculate_risk(
    ocr_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    tampering_result: Dict[str, Any],
    face_result: Dict[str, Any],
    quality_result: Dict[str, Any] = None,
    template_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    score = 0.0
    reasons = []

    # 1. Validation Failures
    if not validation_result.get("valid", True):
        points = min(25.0, 100 - validation_result.get("score", 100))
        if points > 0:
            score += points
            reasons.append({
                "factor": "validation_failure",
                "points": round(points, 2),
                "message": "Document validation checks failed"
            })
    
    # Identify specific critical validation issues
    checks = validation_result.get("checks", [])
    for check in checks:
        if check.get("name") == "Expiry date" and check.get("status") in ("fail", "warn"):
            score += 30.0
            reasons.append({
                "factor": "document_expired",
                "points": 30.0,
                "message": "Document is expired or expiry date is anomalous"
            })
        if check.get("name").startswith("Visual vs MRZ") and check.get("status") in ("fail", "warn"):
            score += 25.0
            reasons.append({
                "factor": "mrz_crosscheck_failure",
                "points": 25.0,
                "message": f"Visual OCR data does not match MRZ data: {check.get('message')}"
            })
        if check.get("name") == "MRZ Composite check" and check.get("status") == "fail":
            score += 30.0
            reasons.append({
                "factor": "mrz_checksum_invalid",
                "points": 30.0,
                "message": "MRZ checksums are invalid"
            })

    # 2. Quality Assessment
    if quality_result:
        if not quality_result.get("acceptable", True):
            points = 20.0
            score += points
            issues_str = ", ".join(quality_result.get("issues", []))
            reasons.append({
                "factor": "poor_image_quality",
                "points": points,
                "message": f"Poor image quality: {issues_str}"
            })

    # 3. Template/Layout Consistency
    if template_result and template_result.get("available"):
        if not template_result.get("layout_consistent"):
            points = 20.0
            score += points
            reasons.append({
                "factor": "layout_anomaly",
                "points": points,
                "message": "Document geometry/layout deviates from standard templates"
            })

    # 4. OCR Confidence
    ocr_conf = ocr_result.get("confidence", 1.0)
    if ocr_conf < 0.6:
        score += 10.0
        reasons.append({
            "factor": "low_ocr_confidence",
            "points": 10.0,
            "message": "OCR text extraction confidence is very low"
        })

    # 5. Tampering
    tamp_score = tampering_result.get("score", 0.0)
    tamp_points = tamp_score * 0.45
    if tamp_points > 0:
        score += tamp_points
        reasons.append({
            "factor": "tampering_detected",
            "points": round(tamp_points, 2),
            "message": f"Document shows signs of digital tampering (score: {tamp_score})"
        })

    # 6. Face Verification
    if face_result.get("available"):
        if face_result.get("match") is False:
            score += 40.0
            reasons.append({
                "factor": "face_mismatch",
                "points": 40.0,
                "message": "Traveller face does not sufficiently match the document portrait"
            })
        elif face_result.get("face_detected_document") is False or face_result.get("face_detected_live") is False:
             score += 20.0
             reasons.append({
                "factor": "face_not_detected",
                "points": 20.0,
                "message": "Face could not be detected in one or both images"
            })

    # Cap at 100
    score = min(100.0, round(score, 2))

    # Risk bands
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
