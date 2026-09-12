# AI-Based Fake Identity & Document Screening System

This repository contains the foundational structure and scaffolding for the Smart India Hackathon (Ministry of Home Affairs / SSB, theme: Blockchain & Cybersecurity).

## Repository Structure

The project maps to the SIH architecture modules as follows:
- `frontend/` - React + Vite + TypeScript application for the Officer UI (Dashboard, Upload).
- `backend/gateway/` - FastAPI API Gateway handling JWT auth, role-based access, and DB management.
- `backend/services/ocr/` - OCR Extraction Module (Mocked).
- `backend/services/validation/` - Document Validation Module (Mocked).
- `backend/services/tamper/` - Tampering Detection Module (Mocked).
- `backend/services/face/` - Face Verification Module (Mocked).
- `backend/services/risk/` - Risk Scoring Engine (Mocked).
- `ml/training/` - Placeholder for AI model training notebooks and scripts.
- `ml/data/` - Placeholder for synthetic dataset generation scripts.
- `infra/` - Infrastructure configuration (currently Docker Compose sits at the root).
- `docs/` - Contains the full `SIH_Project_Plan.txt` for reference.

## Running Locally

### Prerequisites
- Docker and Docker Compose
- Node.js (for frontend dev server)

### 1. Run the Backend & Database
From the root of the project, run:
```bash
docker-compose up --build
```
This will start PostgreSQL, Redis, the API Gateway on port `8000`, and the 5 microservices on ports `8001` through `8005`.

- Gateway Docs: http://localhost:8000/docs
- OCR Docs: http://localhost:8001/docs
- Validation Docs: http://localhost:8002/docs
- Tamper Docs: http://localhost:8003/docs
- Face Docs: http://localhost:8004/docs
- Risk Docs: http://localhost:8005/docs

### 2. Run the Frontend Shell
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The React frontend will be available at http://localhost:5173.

## Note on Implementation
Currently, all AI services return a mock JSON response (including mock explainability payloads) to ensure the REST API shapes are established and the frontend shell is clickable end-to-end. Next steps involve building the actual AI modules (starting with OCR and Validation).
