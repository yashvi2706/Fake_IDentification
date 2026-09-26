"""
Mock services for OCR, validation, and tampering.
Used by Person 4 for independent development.
"""
from typing import Dict, Any

def mock_extract_document_data(image_path: str, document_type: str) -> Dict[str, Any]:
    return {
        "name": "John Doe",
        "document_number": "P1234567",
        "nationality": "IND",
        "date_of_birth": "1995-03-17",
        "date_of_expiry": "2030-04-12",
        "gender": "M",
        "confidence": 0.94
    }

def mock_validate_document(extracted_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
    return {
        "valid": True,
        "score": 92,
        "checks": [
            {
                "name": "Expiry date",
                "status": "pass",
                "message": "Document has not expired"
            }
        ]
    }

def mock_analyze_tampering(image_path: str) -> Dict[str, Any]:
    return {
        "score": 18,
        "suspicious": False,
        "photo_replacement": False,
        "text_manipulation": False,
        "metadata_anomaly": False,
        "compression_anomaly": False,
        "indicators": []
    }
