"""FastAPI wrapper for tamper detection service."""

from typing import Any, Dict, List, Optional
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from tampering_service import analyze_tampering

app = FastAPI(
    title="Tamper Detection Service",
    description="Face-aware tamper detection with TruFor + compact multi-signal model",
    version="3.0.0",
)


class TamperDetails(BaseModel):
    photo_replacement: int
    document_tamper: int = 0
    text_manipulation: int
    metadata_anomaly: int
    compression_anomaly: int

    photo_probability: int = 0
    document_probability: int = 0
    trufor_probability: int = 0

    face_count: int = 0
    no_face_detected: bool = True
    photo_available: bool = False

    trufor_available: bool = False
    trufor_downscaled: bool = False

    model_status: Dict[str, Any] = {}
    face_detector_status: str = "unknown"
    visualization: Optional[str] = None


class TamperResponse(BaseModel):
    score: int
    suspicious: bool
    details: TamperDetails
    indicators: List[str] = []


@app.get("/health")
def health_check():
    from tampering_service import _model_loaded, get_model_status

    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "model_status": get_model_status(),
        "version": "3.0.0",
    }


@app.post("/detect", response_model=TamperResponse)
async def detect_tampering(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    try:
        Image.open(io.BytesIO(img_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=422, detail="Not a valid image file")

    result = await run_in_threadpool(analyze_tampering, img_bytes)
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
