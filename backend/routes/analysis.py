"""
FastAPI routes for analysis.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import uuid
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from schemas.analysis import AnalysisResponse, ErrorDetail
from services.ocr_service import extract_document_data
from services.validation_service import validate_document
from services.tamper.tampering_service import analyze_tampering
from services.face_service import verify_faces
from services.risk_service import calculate_risk
from services.quality_service import analyze_image_quality
from services.template_service import analyze_document_layout

router = APIRouter()

SUPPORTED_DOC_TYPES = {"passport", "visa", "national_id", "driving_license", "permit"}
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorDetail}, 500: {"model": ErrorDetail}}
)
async def analyze_document(
    document: UploadFile = File(...),
    face_image: Optional[UploadFile] = File(None),
    document_type: str = Form(...)
):
    # 1. Validate document type
    doc_type_lower = document_type.lower()
    if doc_type_lower not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported document type. Must be one of: {', '.join(SUPPORTED_DOC_TYPES)}"
        )

    # Validate file extension
    doc_ext = os.path.splitext(document.filename)[1].lower()
    if doc_ext not in SUPPORTED_EXTENSIONS:
         raise HTTPException(
            status_code=400, 
            detail="Unsupported image format. Use JPEG or PNG."
        )

    # 2. Save document temporarily
    doc_id = str(uuid.uuid4())
    doc_path = os.path.join(UPLOAD_DIR, f"doc_{doc_id}{doc_ext}")
    try:
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save document")

    # 3. Save optional traveller face image
    face_path = None
    if face_image:
        face_ext = os.path.splitext(face_image.filename)[1].lower()
        if face_ext not in SUPPORTED_EXTENSIONS:
            os.remove(doc_path)
            raise HTTPException(
                status_code=400, 
                detail="Unsupported face image format. Use JPEG or PNG."
            )
        face_path = os.path.join(UPLOAD_DIR, f"face_{doc_id}{face_ext}")
        try:
            with open(face_path, "wb") as buffer:
                shutil.copyfileobj(face_image.file, buffer)
        except Exception as e:
            os.remove(doc_path)
            raise HTTPException(status_code=500, detail="Failed to save face image")

    try:
        # 4. Call REAL services
        quality_data = analyze_image_quality(doc_path)
        ocr_data = extract_document_data(doc_path, doc_type_lower)
        template_data = analyze_document_layout(doc_path, doc_type_lower)
        validation_data = validate_document(ocr_data, doc_type_lower)
        tampering_data = analyze_tampering(doc_path)
        face_data = verify_faces(doc_path, face_path)
        
        # 5. Risk calculation fuses everything
        risk_data = calculate_risk(
            ocr_data, 
            validation_data, 
            tampering_data, 
            face_data, 
            quality_data, 
            template_data
        )

        screening_id = f"SCR-{str(uuid.uuid4())[:8].upper()}"

        result = {
            "screening_id": screening_id,
            "document_type": doc_type_lower,
            "ocr": ocr_data,
            "validation": validation_data,
            "tampering": tampering_data,
            "face_verification": face_data,
            "risk": risk_data,
            "quality": quality_data,
            "template": template_data
        }
        
        # Log to audit trail
        from services.audit_service import log_screening
        log_screening(result)
        
        return result

    except Exception as e:
        logger.error("Analysis pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {type(e).__name__}: {e}")
    
    finally:
        # 6. Clean temporary files
        if os.path.exists(doc_path):
            os.remove(doc_path)
        if face_path and os.path.exists(face_path):
            os.remove(face_path)

@router.get("/audit")
def get_audit_logs(limit: int = 50):
    from services.audit_service import get_recent_audit_logs
    return {"logs": get_recent_audit_logs(limit)}

