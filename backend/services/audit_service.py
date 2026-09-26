"""
Audit Trail Service.

Logs all document screenings for compliance and monitoring.
Uses SQLite for simple local persistence without external dependencies.
"""
import sqlite3
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DB_PATH = "audit_log.db"

def _init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screening_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                document_type TEXT,
                risk_level TEXT,
                decision TEXT,
                full_result TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize audit DB: {e}")

# Initialize on module import
_init_db()

def log_screening(result: Dict[str, Any]) -> None:
    """Log a completed screening to the audit trail."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO audit_logs (screening_id, document_type, risk_level, decision, full_result)
               VALUES (?, ?, ?, ?, ?)''',
            (
                result.get("screening_id", "UNKNOWN"),
                result.get("document_type", "UNKNOWN"),
                result.get("risk", {}).get("level", "UNKNOWN"),
                result.get("risk", {}).get("decision", "UNKNOWN"),
                json.dumps(result, default=str)
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log screening to audit trail: {e}")

def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent audit logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT screening_id, timestamp, document_type, risk_level, decision 
               FROM audit_logs ORDER BY id DESC LIMIT ?''',
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve audit logs: {e}")
        return []
