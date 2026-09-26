import type { ScreeningResponse } from '../types/screening';
import { analyzeDocumentMock } from './mockApi';

// Configuration to easily switch between mock and real API
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API !== 'false'; // Defaults to true if not explicitly set to false

export const analyzeDocument = async (
  documentFile: File,
  faceImageFile: File | null,
  documentType: string
): Promise<ScreeningResponse> => {
  if (USE_MOCK_API) {
    return analyzeDocumentMock(documentType);
  }

  // Real implementation for future integration
  const formData = new FormData();
  formData.append('document', documentFile);
  if (faceImageFile) {
    formData.append('face_image', faceImageFile);
  }
  formData.append('document_type', documentType);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Analysis failed');
  }

  return response.json();
};
