"""
Pydantic models for the unified analysis API response.

These schemas define the contract between the backend and
Person 1's frontend. Do NOT change the top-level keys of
AnalysisResponse without coordinating with the team.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ── OCR ──────────────────────────────────────────────────────────────

class OCRResult(BaseModel):
    """Fields extracted by the OCR / data-extraction service."""
    name: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_expiry: Optional[str] = None
    gender: Optional[str] = None
    confidence: float = 0.0

    class Config:
        extra = "allow"                       # forward any extra keys from OCR


# ── Validation ───────────────────────────────────────────────────────

class ValidationCheck(BaseModel):
    name: str
    status: str                               # "pass" | "fail" | "warn"
    message: str


class ValidationResult(BaseModel):
    valid: bool = False
    score: int = 0
    checks: List[ValidationCheck] = Field(default_factory=list)


# ── Tampering ────────────────────────────────────────────────────────

class TamperingResult(BaseModel):
    score: float = 0.0
    suspicious: bool = False
    photo_replacement: bool = False
    text_manipulation: bool = False
    metadata_anomaly: bool = False
    compression_anomaly: bool = False
    indicators: List[str] = Field(default_factory=list)


# ── Face Verification ────────────────────────────────────────────────

class FaceVerificationResult(BaseModel):
    available: bool = False
    face_detected_document: Optional[bool] = None
    face_detected_live: Optional[bool] = None
    match: Optional[bool] = None
    similarity: Optional[float] = None
    message: str = ""


# ── Risk ─────────────────────────────────────────────────────────────

class RiskReason(BaseModel):
    factor: str
    points: float
    message: str


class RiskResult(BaseModel):
    score: float = 0.0
    level: str = "LOW"                        # LOW | REVIEW | HIGH
    decision: str = "CLEAR"                   # CLEAR | MANUAL_REVIEW | ESCALATE
    reasons: List[RiskReason] = Field(default_factory=list)


# ── Unified Response ─────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Top-level response returned by POST /api/analyze."""
    screening_id: str
    document_type: str
    ocr: OCRResult
    validation: ValidationResult
    tampering: TamperingResult
    face_verification: FaceVerificationResult
    risk: RiskResult


# ── Error response helper ────────────────────────────────────────────

class ErrorDetail(BaseModel):
    detail: str
