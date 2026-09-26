# AI-Based Fake Identity & Document Screening System

This repository contains the integrated final implementation for the Smart India Hackathon project.

## Modules Included

- **OCR Extraction**: Extracts text and structural data from uploaded documents.
- **Document Validation**: Performs rule-based validation on extracted fields (e.g. expiry checks).
- **Tampering Detection**: Analyzes image metadata, compression, and visual features to identify manipulation.
- **Face Verification**: Compares a traveler's live face image with the face detected in the document.
- **Risk Scoring**: Evaluates the results from the above modules to calculate a unified risk score.

## Final System Structure

- `frontend/` - React + Vite + TypeScript application for the Officer UI (Dashboard, Upload).
- `backend/` - FastAPI backend implementing the core AI analysis modules (`main.py` entrypoint).

## How to Run

### 1. Run the Backend

Open a terminal and set up the backend:

```bash
cd backend
python -m venv .venv
# Activate environment:
# On Windows: .venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The FastAPI backend will be available at http://localhost:8000.
API Documentation: http://localhost:8000/docs

### 2. Run the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The React frontend will be available at http://localhost:5173.

### Required Environment Variables

**Frontend (`frontend/.env`)**:
```
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://localhost:8000
```

### Supported Document Types

- passport
- visa
- national_id
- driving_license
- permit

### Known Limitations

- Tampering detection relies on basic Error Level Analysis (ELA) and ResNet embeddings. Accuracy may vary depending on image quality.
- Face verification relies on the DeepFace VGG-Face model.
- OCR relies on EasyOCR.
- This is an AI-assisted decision support prototype. It does not integrate with any real government databases or actual border watchlists.
