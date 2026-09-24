"""
FastAPI wrapper for the tampering detection service.

This is the HTTP interface. The actual analysis logic lives in tampering_service.py.
Person 4 can either:
  - Call the HTTP endpoint: POST /detect
  - Import directly: from services.tampering_service import analyze_tampering
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional
import io
import os
from PIL import Image

from tampering_service import analyze_tampering

app = FastAPI(
    title="Tamper Detection Service",
    description="Detects physical and digital document tampering using multi-head CNN + EXIF heuristics",
    version="2.0.0",
)


# ─── Response Models ─────────────────────────────────────────────

class TamperDetails(BaseModel):
    photo_replacement: int
    text_manipulation: int
    metadata_anomaly: int
    compression_anomaly: int


class TamperResponse(BaseModel):
    score: int
    suspicious: bool
    details: TamperDetails


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint for Docker/orchestrator probes."""
    from tampering_service import _model_loaded
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "version": "2.0.0",
    }


@app.post("/detect", response_model=TamperResponse)
async def detect_tampering(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for tampering indicators.

    Accepts: JPEG, PNG, BMP, TIFF, WebP
    Returns: Deterministic score (0-100), suspicious flag, sub-scores, and indicators.
    """
    # Read file bytes
    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Reject oversized files (>15 MB)
    if len(img_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    # Reject non-image files before hitting the model
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=422, detail="Not a valid image file")

    # Run CPU-bound analysis off the async event loop so health checks don't stall
    result = await run_in_threadpool(analyze_tampering, img_bytes)

    return result


# ─── Startup ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
