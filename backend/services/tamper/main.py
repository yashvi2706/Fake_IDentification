from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Tamper Detection Service", description="Detects physical and digital document tampering")

class TamperResponse(BaseModel):
    status: str
    score: float
    flags: List[str]
    explainability: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/detect", response_model=TamperResponse)
def detect_tampering(file: UploadFile = File(...)):
    # Mock Tamper Detection
    return {
        "status": "warning",
        "score": 0.76,
        "flags": ["Possible copy-move forgery on photo", "Metadata anomaly"],
        "explainability": {
            "ela_max_val": 185.3,
            "metadata_software_signature": "Adobe Photoshop",
            "heatmap_grid": [[0.1, 0.2], [0.8, 0.9]] # Mock heatmap array
        }
    }
