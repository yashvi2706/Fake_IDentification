from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Validation Service", description="Rule engine and MRZ validation")

class ValidationRequest(BaseModel):
    extracted_data: dict

class ValidationResponse(BaseModel):
    status: str
    issues: List[str]
    explainability: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/validate", response_model=ValidationResponse)
def validate_data(req: ValidationRequest):
    # Mock Validation
    return {
        "status": "failed",
        "issues": ["MRZ checksum mismatch"],
        "explainability": {
            "rules_evaluated": 12,
            "failed_rules": ["mrz_checksum"]
        }
    }
