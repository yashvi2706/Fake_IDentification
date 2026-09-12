from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="Face Verification Service", description="Matches faces from documents against live captures")

class FaceResponse(BaseModel):
    status: str
    match_score: float
    match: bool
    explainability: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify", response_model=FaceResponse)
def verify_face(document_face: UploadFile = File(...), live_face: UploadFile = File(...)):
    # Mock Face Match
    return {
        "status": "success",
        "match_score": 0.88,
        "match": True,
        "explainability": {
            "document_face_detected": True,
            "live_face_detected": True,
            "liveness_score": 0.92
        }
    }
