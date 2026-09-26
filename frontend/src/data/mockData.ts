import type { ScreeningSummary } from '../types/screening';

export const mockRecentScreenings: ScreeningSummary[] = [
  {
    id: "SCR-A1B2C3D4",
    document_type: "Passport",
    person_name: "John Doe",
    risk_level: "LOW",
    decision: "CLEAR",
    timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
  },
  {
    id: "SCR-F9E8D7C6",
    document_type: "National ID",
    person_name: "Jane Smith",
    risk_level: "REVIEW",
    decision: "MANUAL_REVIEW",
    timestamp: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
  },
  {
    id: "SCR-X5Y6Z7W8",
    document_type: "Driving License",
    person_name: "Robert Johnson",
    risk_level: "HIGH",
    decision: "ESCALATE",
    timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
  },
  {
    id: "SCR-M1N2P3Q4",
    document_type: "Passport",
    person_name: "Emily Davis",
    risk_level: "LOW",
    decision: "CLEAR",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
  }
];

export const mockStats = {
  totalScreened: 1248,
  lowRisk: 1102,
  flagged: 146,
  avgTime: "2.4s"
};
