import type { ScreeningResponse } from '../types/screening';
import { analyzeDocumentMock } from './mockApi';

// USE_MOCK_API is false unless explicitly set to 'true'
// IMPORTANT: Vite bakes env vars at BUILD TIME.
// If VITE_USE_MOCK_API is not set in Vercel → import.meta.env.VITE_USE_MOCK_API is undefined → false → real API
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === 'true';

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
    // Read error body for debugging
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

  return response.json();
};

