export interface OCRResults {
  name?: string | null;
  document_number?: string | null;
  nationality?: string | null;
  date_of_birth?: string | null;
  date_of_expiry?: string | null;
  gender?: string | null;
  confidence: number;
  [key: string]: any;
}

export interface ValidationCheck {
  name: string;
  status: 'pass' | 'fail' | 'warn' | 'warning';
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
  face_detected_document?: boolean | null;
  face_detected_live?: boolean | null;
  match?: boolean | null;
  similarity?: number | null;
  message: string;
}

export interface RiskReason {
  factor: string;
  points: number;
  message: string;
}

export interface RiskFactors {
  score: number;
  level: 'LOW' | 'REVIEW' | 'HIGH';
  decision: 'CLEAR' | 'MANUAL_REVIEW' | 'ESCALATE';
  reasons: RiskReason[];  // objects, not strings
}

export interface QualityResults {
  score: number;
  acceptable: boolean;
  blur_score: number;
  glare_score: number;
  brightness_score: number;
  resolution_ok: boolean;
  document_visible: boolean;
  issues: string[];
}

export interface TemplateResults {
  available: boolean;
  score: number;
  layout_consistent?: boolean | null;
  checks: ValidationCheck[];
}

export interface ScreeningResponse {
  screening_id: string;
  document_type: string;
  ocr: OCRResults;
  validation: ValidationResults;
  tampering: TamperingResults;
  face_verification: FaceVerification;
  risk: RiskFactors;
  quality?: QualityResults;
  template?: TemplateResults;
}

export interface ScreeningSummary {
  id: string;
  document_type: string;
  person_name: string;
  risk_level: 'LOW' | 'REVIEW' | 'HIGH';
  decision: 'CLEAR' | 'MANUAL_REVIEW' | 'ESCALATE';
  timestamp: string;
}

