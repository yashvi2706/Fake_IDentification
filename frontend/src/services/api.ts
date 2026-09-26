import type { ScreeningResponse, RiskReason } from '../types/screening';
import { analyzeDocumentMock } from './mockApi';

// USE_MOCK_API is false unless explicitly set to 'true'
// IMPORTANT: Vite bakes env vars at BUILD TIME.
// If VITE_USE_MOCK_API is not set in Vercel → import.meta.env.VITE_USE_MOCK_API is undefined → false → real API
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === 'true';

/**
 * Normalize the raw backend response to guarantee safe defaults for every field.
 * Prevents React render crashes from null arrays, wrong numeric formats, etc.
 */
function normalizeAnalysisResponse(raw: any): ScreeningResponse {
  // Normalize risk reasons: backend sends [{factor,points,message}], type was wrongly string[]
  const rawReasons: any[] = Array.isArray(raw?.risk?.reasons) ? raw.risk.reasons : [];
  const reasons: RiskReason[] = rawReasons.map((r: any) =>
    typeof r === 'string'
      ? { factor: 'unknown', points: 0, message: r }
      : { factor: r?.factor ?? 'unknown', points: Number(r?.points ?? 0), message: r?.message ?? String(r) }
  );

  // Normalize OCR confidence: backend may return 0-1 or 0-100
  const rawConf = raw?.ocr?.confidence ?? 0;
  const confidence = rawConf > 1 ? rawConf / 100 : rawConf;

  return {
    screening_id: raw?.screening_id ?? 'UNKNOWN',
    document_type: raw?.document_type ?? 'unknown',
    ocr: {
      ...(raw?.ocr ?? {}),
      confidence,
    },
    validation: {
      valid: raw?.validation?.valid ?? false,
      score: raw?.validation?.score ?? 0,
      checks: Array.isArray(raw?.validation?.checks) ? raw.validation.checks : [],
    },
    tampering: {
      score: raw?.tampering?.score ?? 0,
      suspicious: raw?.tampering?.suspicious ?? false,
      photo_replacement: raw?.tampering?.photo_replacement ?? false,
      text_manipulation: raw?.tampering?.text_manipulation ?? false,
      metadata_anomaly: raw?.tampering?.metadata_anomaly ?? false,
      compression_anomaly: raw?.tampering?.compression_anomaly ?? false,
      indicators: Array.isArray(raw?.tampering?.indicators) ? raw.tampering.indicators : [],
    },
    face_verification: {
      available: raw?.face_verification?.available ?? false,
      face_detected_document: raw?.face_verification?.face_detected_document ?? null,
      face_detected_live: raw?.face_verification?.face_detected_live ?? null,
      match: raw?.face_verification?.match ?? null,
      similarity: raw?.face_verification?.similarity ?? null,
      message: raw?.face_verification?.message ?? '',
    },
    risk: {
      score: raw?.risk?.score ?? 0,
      level: raw?.risk?.level ?? 'LOW',
      decision: raw?.risk?.decision ?? 'CLEAR',
      reasons,
    },
  };
}

export const analyzeDocument = async (
  documentFile: File,
  faceImageFile: File | null,
  documentType: string
): Promise<ScreeningResponse> => {
  if (USE_MOCK_API) {
    return analyzeDocumentMock(documentType);
  }

  const formData = new FormData();
  formData.append('document', documentFile);
  if (faceImageFile) {
    formData.append('face_image', faceImageFile);
  }
  formData.append('document_type', documentType);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
  const url = `${API_BASE}/api/analyze`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
  } catch (networkErr) {
    // Network-level failure: no connection, CORS preflight blocked, etc.
    console.error('[API] Network error calling', url, networkErr);
    throw new Error(`Backend unreachable: ${networkErr instanceof Error ? networkErr.message : networkErr}`);
  }

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson?.detail ?? errorDetail;
    } catch {
      try {
        const errText = await response.text();
        if (errText) errorDetail = errText.slice(0, 200);
      } catch { /* ignore */ }
    }
    console.error('[API] Request failed:', response.status, errorDetail);
    throw new Error(errorDetail);
  }

  const raw = await response.json();
  console.log('[API] Raw response:', raw);
  return normalizeAnalysisResponse(raw);
};


