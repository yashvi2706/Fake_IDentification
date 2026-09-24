import type { ScreeningResponse } from '../types/screening';

export const analyzeDocumentMock = async (documentType: string): Promise<ScreeningResponse> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        screening_id: `SCR-${Math.random().toString(36).substring(2, 10).toUpperCase()}`,
        document_type: documentType,
        ocr: {
          name: "John Doe",
          document_number: "P1234567",
          nationality: "IND",
          date_of_birth: "1995-03-17",
          date_of_expiry: "2030-04-12",
          gender: "M",
          confidence: 0.94
        },
        validation: {
          valid: true,
          score: 92,
          checks: [
            {
              name: "Expiry date",
              status: "pass",
              message: "Document has not expired"
            },
            {
              name: "Format structure",
              status: "pass",
              message: "MRZ structure is valid"
            }
          ]
        },
        tampering: {
          score: 18,
          suspicious: false,
          photo_replacement: false,
          text_manipulation: false,
          metadata_anomaly: false,
          compression_anomaly: false,
          indicators: []
        },
        face_verification: {
          available: true,
          face_detected_document: true,
          face_detected_live: true,
          match: true,
          similarity: 0.91,
          message: "Faces appear to match"
        },
        risk: {
          score: 14,
          level: "LOW",
          decision: "CLEAR",
          reasons: []
        }
      });
    }, 2500); // Simulate network and processing delay
  });
};
