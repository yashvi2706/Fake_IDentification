export interface OCRResults {
  name: string;
  document_number: string;
  nationality: string;
  date_of_birth: string;
  date_of_expiry: string;
  gender: string;
  confidence: number;
  [key: string]: any; // For flexible visa fields
}

export interface ValidationCheck {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
}

export interface ValidationResults {
  valid: boolean;
  score: number;
  checks: ValidationCheck[];
}

export interface TamperingResults {
  score: number;
  suspicious: boolean;
  photo_replacement: boolean;
  text_manipulation: boolean;
  metadata_anomaly: boolean;
  compression_anomaly: boolean;
  indicators: string[];
}

export interface FaceVerification {
  available: boolean;
  face_detected_document: boolean;
  face_detected_live: boolean;
  match: boolean;
  similarity: number;
  message: string;
}

export interface RiskFactors {
  score: number;
  level: 'LOW' | 'REVIEW' | 'HIGH';
  decision: 'CLEAR' | 'MANUAL_REVIEW' | 'ESCALATE';
  reasons: string[];
}

export interface ScreeningResponse {
  screening_id: string;
  document_type: string;
  ocr: OCRResults;
  validation: ValidationResults;
  tampering: TamperingResults;
  face_verification: FaceVerification;
  risk: RiskFactors;
}

export interface ScreeningSummary {
  id: string;
  document_type: string;
  person_name: string;
  risk_level: 'LOW' | 'REVIEW' | 'HIGH';
  decision: 'CLEAR' | 'MANUAL_REVIEW' | 'ESCALATE';
  timestamp: string;
}
