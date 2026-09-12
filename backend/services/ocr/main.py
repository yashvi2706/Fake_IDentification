from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import time

app = FastAPI(title="OCR Service", description="Extracts text and fields from documents")

class OCRResponse(BaseModel):
    status: str
    confidence: float
    extracted_data: dict
    explainability: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/extract", response_model=OCRResponse)
def extract_document(file: UploadFile = File(...)):
    # Mock OCR processing
    return {
        "status": "success",
        "confidence": 0.94,
        "extracted_data": {
            "document_type": "Passport",
            "document_number": "A1234567",
            "name": "JOHN DOE",
            "nationality": "GBR",
            "dob": "1985-04-12"
        },
        "explainability": {
            "layout_boxes_found": 15,
            "mrz_detected": True
        }
    }
