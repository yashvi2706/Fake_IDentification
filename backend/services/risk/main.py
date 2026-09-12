from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Risk Scoring Engine", description="Fuses results from all modules into a single explainable risk score")

class RiskRequest(BaseModel):
    ocr_results: dict
    validation_results: dict
    tamper_results: dict
    face_results: dict

class RiskResponse(BaseModel):
    risk_score: int
    risk_band: str
    explainability: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/score", response_model=RiskResponse)
def compute_risk(req: RiskRequest):
    # Mock Risk Fusion
    return {
        "risk_score": 82,
        "risk_band": "High",
        "explainability": {
            "top_contributors": ["validation_failure", "tamper_score"],
            "weighted_breakdown": {
                "ocr": 5,
                "validation": 40,
                "tamper": 35,
                "face": 2
            }
        }
    }
