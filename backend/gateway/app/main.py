from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import auth, models
import os

app = FastAPI(title="Gateway API - Document Screening")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "gateway"}

@app.post("/login")
def login(form_data: dict):
    # Mock login for demo scaffolding
    if form_data.get("username") == "admin":
        access_token = auth.create_access_token(data={"sub": "admin", "role": "Admin"})
        return {"access_token": access_token, "token_type": "bearer"}
    elif form_data.get("username") == "officer":
        access_token = auth.create_access_token(data={"sub": "officer", "role": "Officer"})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

# Mock Proxy endpoints for the sub-services (until we use httpx/requests to forward them)
@app.post("/api/screen")
def screen_document(token: str = Depends(auth.get_current_user)):
    # This endpoint would typically coordinate calling OCR, Validation, Tamper, Face, and Risk
    # For scaffolding, returning the mock expected payload.
    return {
        "risk_score": 82,
        "risk_band": "High",
        "modules": {
            "ocr": {"status": "success", "confidence": 0.94},
            "validation": {"status": "failed", "issues": ["MRZ checksum mismatch"]},
            "tamper": {"status": "warning", "score": 0.76, "flags": ["Possible copy-move forgery"]},
            "face": {"status": "success", "match_score": 0.88, "match": True}
        }
    }
