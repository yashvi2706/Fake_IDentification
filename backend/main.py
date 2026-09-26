"""
Main FastAPI application entrypoint for Person 4 backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes import analysis

load_dotenv()

app = FastAPI(
    title="Fake IDentification Backend",
    description="Backend service for face verification and risk scoring.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])

@app.get("/api/health", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
