"""
OCR Extraction Service for Identity Documents.

Provides the extract_document_data() function that processes identity
document images using EasyOCR with OpenCV preprocessing, extracting
structured fields from passports, visas, national IDs, driving licenses,
and permits.

Person 2 — Hackathon Module 1
"""

import os
import re
import logging
from typing import Dict, Optional, Any, List, Tuple

import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Resolve imports whether run from backend/ or from project root
try:
    from utils.date_utils import parse_date, normalize_date
except ImportError:
    from backend.utils.date_utils import parse_date, normalize_date


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level EasyOCR reader (lazy-initialized to avoid startup cost)
# ---------------------------------------------------------------------------
_reader: Optional[Any] = None


def _get_reader() -> Any:
    """Lazy-initialize the EasyOCR reader singleton."""
    global _reader
    if _reader is None:
        if not EASYOCR_AVAILABLE:
            raise RuntimeError(
                "easyocr is not installed. Install it with: pip install easyocr"
            )
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# ===================================================================
#  PUBLIC API
# ===================================================================

SUPPORTED_DOCUMENT_TYPES = [
    "passport",
    "visa",
    "national_id",
    "driving_license",
    "permit",
]


def extract_document_data(image_path: str, document_type: str) -> Dict[str, Any]:
    """
    Extract structured data from an identity document image.

    Args:
        image_path: Absolute or relative path to the document image.
        document_type: One of 'passport', 'visa', 'national_id',
                       'driving_license', 'permit'.

    Returns:
        A dict containing extracted fields, raw_text, and confidence.
        Fields vary by document_type. Missing values are None.
    """
    document_type = document_type.strip().lower() if document_type else ""

    if document_type not in SUPPORTED_DOCUMENT_TYPES:
        logger.warning("Unsupported document type: %s — treating as generic ID", document_type)
        document_type = "national_id"

    # ------------------------------------------------------------------
    # Step 1-5: Load & preprocess
    # ------------------------------------------------------------------
    try:
        preprocessed = _preprocess_image(image_path)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc)
        return _empty_result(document_type, error=str(exc))

    # ------------------------------------------------------------------
    # Step 6: Run OCR
    # ------------------------------------------------------------------
    try:
        ocr_results = _run_ocr(preprocessed)
    except Exception as exc:
        logger.error("OCR engine failed: %s", exc)
        return _empty_result(document_type, error=str(exc))

    # ------------------------------------------------------------------
    # Step 7: Build raw text and compute confidence
    # ------------------------------------------------------------------
    raw_text = _build_raw_text(ocr_results)
    confidence = _compute_confidence(ocr_results)

    # ------------------------------------------------------------------
    # Step 8: Extract fields based on document type
    # ------------------------------------------------------------------
    if document_type == "passport":
        fields = _extract_passport_fields(raw_text, ocr_results)
    elif document_type == "visa":
        fields = _extract_visa_fields(raw_text, ocr_results)
    else:
        # national_id, driving_license, permit
        fields = _extract_generic_id_fields(raw_text, ocr_results)

    fields["raw_text"] = raw_text
    fields["confidence"] = round(confidence, 4)
    return fields


# ===================================================================
#  IMAGE PREPROCESSING
# ===================================================================

def _preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a document image for OCR."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Load
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Step 2: Resize if too large (keep aspect ratio, max width 2000px)
    h, w = img.shape[:2]
    max_width = 2000
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    # Step 3: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 5: Light Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced


# ===================================================================
#  OCR ENGINE
# ===================================================================

def _run_ocr(image: np.ndarray) -> List[Tuple]:
    """
    Run EasyOCR on a preprocessed image.

    Returns list of (bbox, text, confidence) tuples.
    """
    reader = _get_reader()
    results = reader.readtext(image, detail=1, paragraph=False)
    return results


def _build_raw_text(ocr_results: List[Tuple]) -> str:
    """Concatenate OCR detections into a single cleaned string."""
    lines: List[str] = []
    for item in ocr_results:
        text = item[1] if len(item) > 1 else ""
        text = text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _compute_confidence(ocr_results: List[Tuple]) -> float:
    """Weighted average confidence across all detections."""
    if not ocr_results:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for item in ocr_results:
        conf = item[2] if len(item) > 2 else 0.0
        text = item[1] if len(item) > 1 else ""
        weight = len(text)  # weight by text length
        weighted_sum += conf * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


# ===================================================================
#  PASSPORT EXTRACTION
# ===================================================================

def _extract_passport_fields(raw_text: str, ocr_results: List[Tuple]) -> Dict[str, Any]:
    """Extract passport-specific fields from OCR output."""
    result = {
        "name": None,
        "document_number": None,
        "nationality": None,
        "date_of_birth": None,
        "date_of_expiry": None,
        "gender": None,
    }

    # Try MRZ first — it's the most reliable data source on passports
    mrz_data = _parse_mrz(raw_text)
    if mrz_data:
        result.update({k: v for k, v in mrz_data.items() if v is not None})

    # Fill remaining gaps from visual OCR text using label matching
    _fill_from_labels(result, raw_text, document_type="passport")

    # Normalize dates
    for date_field in ("date_of_birth", "date_of_expiry"):
        if result.get(date_field):
            normalized = normalize_date(result[date_field])
            if normalized:
                result[date_field] = normalized

    return result


# ===================================================================
#  VISA EXTRACTION
# ===================================================================

def _extract_visa_fields(raw_text: str, ocr_results: List[Tuple]) -> Dict[str, Any]:
    """Extract visa-specific fields from OCR output."""
    result = {
        "visa_number": None,
        "visa_type": None,
        "valid_from": None,
        "valid_until": None,
        "stay_duration": None,
    }

    lines = raw_text.split("\n")
    text_upper = raw_text.upper()

    # Visa number
    result["visa_number"] = _search_field(
        lines, text_upper,
        labels=["VISA NO", "VISA NUMBER", "VISA #", "NO.", "NUMBER"],
        pattern=r'[A-Z]{0,3}\d{5,12}'
    )

    # Visa type
    result["visa_type"] = _search_field(
        lines, text_upper,
        labels=["TYPE", "VISA TYPE", "CATEGORY", "CLASS"],
        pattern=None
    )

    # Valid from
    result["valid_from"] = _search_date_field(
        lines, text_upper,
        labels=["VALID FROM", "FROM", "ISSUE DATE", "DATE OF ISSUE", "ISSUED"]
    )

    # Valid until
    result["valid_until"] = _search_date_field(
        lines, text_upper,
        labels=["VALID UNTIL", "VALID TO", "EXPIRY", "EXPIRES", "DATE OF EXPIRY",
                "EXPIRATION", "UNTIL"]
    )

    # Stay duration
    stay = _search_field(
        lines, text_upper,
        labels=["DURATION", "STAY", "DURATION OF STAY", "PERIOD", "MAX STAY"],
        pattern=r'\d+\s*(?:DAYS?|MONTHS?|D|M)'
    )
    if stay:
        result["stay_duration"] = stay.strip()

    # Normalize dates
    for date_field in ("valid_from", "valid_until"):
        if result.get(date_field):
            normalized = normalize_date(result[date_field])
            if normalized:
                result[date_field] = normalized

    return result


# ===================================================================
#  GENERIC ID EXTRACTION  (national_id, driving_license, permit)
# ===================================================================

def _extract_generic_id_fields(raw_text: str, ocr_results: List[Tuple]) -> Dict[str, Any]:
    """Extract fields from national IDs, driving licenses, permits."""
    result = {
        "name": None,
        "document_number": None,
        "date_of_birth": None,
        "date_of_expiry": None,
        "address": None,
    }

    lines = raw_text.split("\n")
    text_upper = raw_text.upper()

    # Name
    result["name"] = _search_field(
        lines, text_upper,
        labels=["NAME", "FULL NAME", "SURNAME", "GIVEN NAME", "FIRST NAME"],
        pattern=None
    )

    # Document number
    result["document_number"] = _search_field(
        lines, text_upper,
        labels=["NO", "NUMBER", "DOCUMENT NO", "LICENSE NO", "DL NO",
                "ID NO", "PERMIT NO", "CARD NO"],
        pattern=r'[A-Z0-9]{5,15}'
    )

    # DOB
    result["date_of_birth"] = _search_date_field(
        lines, text_upper,
        labels=["DOB", "DATE OF BIRTH", "BIRTH", "BORN", "D.O.B"]
    )

    # Expiry
    result["date_of_expiry"] = _search_date_field(
        lines, text_upper,
        labels=["EXPIRY", "EXPIRES", "VALID UNTIL", "DATE OF EXPIRY",
                "EXP", "VALID THRU", "EXPIRATION"]
    )

    # Address
    result["address"] = _search_field(
        lines, text_upper,
        labels=["ADDRESS", "ADDR", "RESIDENCE"],
        pattern=None
    )

    # Normalize dates
    for date_field in ("date_of_birth", "date_of_expiry"):
        if result.get(date_field):
            normalized = normalize_date(result[date_field])
            if normalized:
                result[date_field] = normalized

    return result


# ===================================================================
#  MRZ PARSING (TD3 — Passport)
# ===================================================================

def _parse_mrz(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to detect and parse MRZ lines from raw OCR text.

    Handles TD3 passport format (2 lines of 44 characters each).
    Returns extracted fields or None if MRZ is not found.
    """
    try:
        # Clean up OCR artifacts in potential MRZ text
        mrz_candidate = raw_text.replace(" ", "")

        # Find potential MRZ lines (44 chars of A-Z, 0-9, <)
        mrz_pattern = re.compile(r'[A-Z0-9<]{40,50}')
        matches = mrz_pattern.findall(mrz_candidate)

        # Normalize each match to exactly 44 chars
        mrz_lines = []
        for m in matches:
            # Replace common OCR misreads in MRZ
            m = m.replace("O", "0").replace("I", "1").replace("S", "5")
            # Only revert letters that are clearly part of alpha fields
            cleaned = m[:44] if len(m) >= 44 else m
            if len(cleaned) >= 42:  # allow slight tolerance
                cleaned = cleaned.ljust(44, "<")
                mrz_lines.append(cleaned)

        if len(mrz_lines) < 2:
            return None

        line1 = mrz_lines[-2]  # second-to-last (type + name)
        line2 = mrz_lines[-1]  # last (data line)

        result = {}

        # --- Line 1: P<NATIONALITY<<SURNAME<<GIVEN<NAMES ---
        if line1[0] == "P":
            # Nationality: positions 2-5
            nat = line1[2:5].replace("<", "").strip()
            if nat and nat.isalpha():
                result["nationality"] = nat

            # Name: positions 5-44
            name_part = line1[5:44]
            parts = name_part.split("<<")
            surname = parts[0].replace("<", " ").strip() if len(parts) > 0 else ""
            given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            full_name = f"{given} {surname}".strip()
            if full_name:
                result["name"] = full_name

        # --- Line 2: Document#, DOB, Gender, Expiry ---
        # Positions: 0-8 document number, 9 check digit,
        #           10-12 nationality, 13-18 DOB(YYMMDD), 19 check,
        #           20 gender, 21-26 expiry(YYMMDD), 27 check
        doc_num = line2[0:9].replace("<", "").strip()
        if doc_num:
            result["document_number"] = doc_num

        # Verify document number checksum
        doc_check = line2[9] if len(line2) > 9 else None
        if doc_check and doc_check.isdigit():
            expected = _mrz_checksum(line2[0:9])
            if expected is not None and str(expected) != doc_check:
                logger.debug("MRZ document number checksum mismatch")

        # Nationality from line 2 (backup)
        if "nationality" not in result:
            nat2 = line2[10:13].replace("<", "").strip()
            if nat2 and nat2.isalpha():
                result["nationality"] = nat2

        # Date of birth (YYMMDD at positions 13-18)
        dob_raw = line2[13:19]
        dob = _parse_mrz_date(dob_raw, is_birth=True)
        if dob:
            result["date_of_birth"] = dob

        # Gender (position 20)
        gender_char = line2[20] if len(line2) > 20 else ""
        if gender_char in ("M", "F"):
            result["gender"] = gender_char
        elif gender_char == "<" or gender_char == "X":
            result["gender"] = "X"

        # Expiry (YYMMDD at positions 21-26)
        exp_raw = line2[21:27]
        exp = _parse_mrz_date(exp_raw, is_birth=False)
        if exp:
            result["date_of_expiry"] = exp

        return result if result else None

    except Exception as exc:
        logger.debug("MRZ parsing failed (non-fatal): %s", exc)
        return None


def _parse_mrz_date(yymmdd: str, is_birth: bool = True) -> Optional[str]:
    """Convert YYMMDD MRZ date to ISO format."""
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    try:
        yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        if is_birth:
            year = 1900 + yy if yy > 30 else 2000 + yy
        else:
            year = 2000 + yy if yy < 70 else 1900 + yy
        from datetime import date as dt_date
        return dt_date(year, mm, dd).isoformat()
    except (ValueError, OverflowError):
        return None


def _mrz_checksum(data: str) -> Optional[int]:
    """
    Compute ICAO 9303 MRZ check digit.

    Character weights cycle: 7, 3, 1
    A-Z → 10-35, 0-9 → 0-9, < → 0
    """
    try:
        weights = [7, 3, 1]
        total = 0
        for i, ch in enumerate(data):
            if ch.isdigit():
                val = int(ch)
            elif ch.isalpha():
                val = ord(ch.upper()) - ord("A") + 10
            elif ch == "<":
                val = 0
            else:
                val = 0
            total += val * weights[i % 3]
        return total % 10
    except Exception:
        return None


# ===================================================================
#  LABEL & PATTERN MATCHING HELPERS
# ===================================================================

def _fill_from_labels(result: Dict, raw_text: str, document_type: str) -> None:
    """Fill missing fields from OCR text using label-based matching."""
    lines = raw_text.split("\n")
    text_upper = raw_text.upper()

    # Name
    if not result.get("name"):
        result["name"] = _search_field(
            lines, text_upper,
            labels=["SURNAME", "NAME", "GIVEN NAME", "FULL NAME", "FIRST NAME"],
            pattern=None
        )

    # Document number
    if not result.get("document_number"):
        result["document_number"] = _search_field(
            lines, text_upper,
            labels=["PASSPORT NO", "DOCUMENT NO", "NO.", "NUMBER"],
            pattern=r'[A-Z]{1,2}\d{6,8}'
        )

    # Nationality
    if not result.get("nationality"):
        result["nationality"] = _search_field(
            lines, text_upper,
            labels=["NATIONALITY", "COUNTRY", "CITIZEN"],
            pattern=r'[A-Z]{2,3}'
        )

    # DOB
    if not result.get("date_of_birth"):
        result["date_of_birth"] = _search_date_field(
            lines, text_upper,
            labels=["DOB", "DATE OF BIRTH", "BIRTH", "BORN", "D.O.B"]
        )

    # Expiry
    if not result.get("date_of_expiry"):
        result["date_of_expiry"] = _search_date_field(
            lines, text_upper,
            labels=["EXPIRY", "EXPIRATION", "DATE OF EXPIRY", "EXP", "VALID UNTIL"]
        )

    # Gender
    if not result.get("gender"):
        gender = _search_field(
            lines, text_upper,
            labels=["SEX", "GENDER"],
            pattern=r'[MFX](?:\b|ALE|EMALE)?'
        )
        if gender:
            g = gender.strip().upper()
            if g.startswith("M"):
                result["gender"] = "M"
            elif g.startswith("F"):
                result["gender"] = "F"
            else:
                result["gender"] = g


def _search_field(
    lines: List[str],
    text_upper: str,
    labels: List[str],
    pattern: Optional[str],
) -> Optional[str]:
    """
    Search OCR lines for a field value following a label.

    Strategy:
    1. Look for label on the same line, take the part after it.
    2. If label found alone on a line, take the next line.
    3. If pattern given, search globally for first match.
    """
    for label in labels:
        label_upper = label.upper()
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()

            # Check if label appears in this line
            idx = line_upper.find(label_upper)
            if idx == -1:
                continue

            # Value after label on the same line
            after = line[idx + len(label):].strip()
            # Remove common separators
            after = re.sub(r'^[\s:;\-/]+', '', after).strip()

            if after and len(after) >= 1:
                return after

            # Value on the next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) >= 1:
                    return next_line

    # Fallback: global pattern search
    if pattern:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(0)

    return None


def _search_date_field(
    lines: List[str],
    text_upper: str,
    labels: List[str],
) -> Optional[str]:
    """Search for a date value associated with a label."""
    # Date pattern for inline search
    date_pattern = r'\d{1,4}[\s/\-\.]\w{1,9}[\s/\-\.]\d{2,4}'

    for label in labels:
        label_upper = label.upper()
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            idx = line_upper.find(label_upper)
            if idx == -1:
                continue

            # Look for a date after the label on the same line
            remainder = line[idx + len(label):]
            match = re.search(date_pattern, remainder)
            if match:
                return match.group(0).strip()

            # Try the next line
            if i + 1 < len(lines):
                match = re.search(date_pattern, lines[i + 1])
                if match:
                    return match.group(0).strip()

    return None


# ===================================================================
#  EMPTY / ERROR RESULT TEMPLATES
# ===================================================================

def _empty_result(document_type: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Return an empty result dict appropriate for the document type."""
    if document_type == "passport":
        result = {
            "name": None,
            "document_number": None,
            "nationality": None,
            "date_of_birth": None,
            "date_of_expiry": None,
            "gender": None,
        }
    elif document_type == "visa":
        result = {
            "visa_number": None,
            "visa_type": None,
            "valid_from": None,
            "valid_until": None,
            "stay_duration": None,
        }
    else:
        result = {
            "name": None,
            "document_number": None,
            "date_of_birth": None,
            "date_of_expiry": None,
            "address": None,
        }

    result["raw_text"] = ""
    result["confidence"] = 0.0
    if error:
        result["error"] = error
    return result
