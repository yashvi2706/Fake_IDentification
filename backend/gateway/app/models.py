import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, JSON, Enum, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import enum

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sih_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class RoleEnum(str, enum.Enum):
    OFFICER = "Officer"
    ADMIN = "Admin"
    AUDITOR = "Auditor"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.OFFICER)
    created_at = Column(DateTime, default=datetime.utcnow)

class Traveler(Base):
    __tablename__ = "travelers"
    id = Column(Integer, primary_key=True, index=True)
    primary_name = Column(String(150))
    nationality = Column(String(10))
    dob = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    documents = relationship("Document", back_populates="traveler")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    traveler_id = Column(Integer, ForeignKey("travelers.id"))
    document_type = Column(String(50)) # Passport, Visa, etc.
    document_number = Column(String(50), index=True)
    issue_date = Column(DateTime)
    expiry_date = Column(DateTime)
    traveler = relationship("Traveler", back_populates="documents")
    scans = relationship("Scan", back_populates="document")

class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    image_path = Column(String(255))
    scanned_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="scans")
    screening_decision = relationship("ScreeningDecision", back_populates="scan", uselist=False)

class ScreeningDecision(Base):
    __tablename__ = "screening_decisions"
    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), unique=True)
    officer_id = Column(Integer, ForeignKey("users.id"))
    risk_score = Column(Integer)
    risk_band = Column(String(20)) # Low, Medium, High
    reasoning_payload = Column(JSON) # Store confidence scores and explainability here
    decision_time = Column(DateTime, default=datetime.utcnow)
    scan = relationship("Scan", back_populates="screening_decision")

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, ForeignKey("screening_decisions.id"), nullable=True)
    action = Column(String(100)) # "SCREENING_COMPLETED", "MANUAL_OVERRIDE"
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    previous_hash = Column(String(256), nullable=False) # For hash-chaining
    current_hash = Column(String(256), nullable=False)
