"""
Document Validation Service for Identity Documents.

Provides the validate_document() function that performs structural and
data-consistency checks on OCR-extracted document data. Checks include
field presence, date validity, format verification, and MRZ consistency.

IMPORTANT LIMITATION:
Validation only checks structural and data consistency. A forged document
may still contain structurally valid information. Results should be
interpreted as "Document fields are structurally consistent" — NOT as
"This document is officially genuine."

Person 2 — Hackathon Module 2
"""

import re
import logging
from datetime import date
from typing import Dict, Any, List, Optional

# Resolve imports whether run from backend/ or from project root
try:
    from utils.date_utils import parse_date, is_expired, is_date_reasonable
except ImportError:
    from backend.utils.date_utils import parse_date, is_expired, is_date_reasonable


logger = logging.getLogger(__name__)


# ===================================================================
#  PUBLIC API
# ===================================================================

def validate_document(extracted_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
    """
    Validate extracted document data for structural and data consistency.

    Args:
        extracted_data: Dict returned by extract_document_data().
        document_type: One of 'passport', 'visa', 'national_id',
                       'driving_license', 'permit'.

    Returns:
        {
            "valid": bool,
            "score": int (0-100),
            "checks": [
                {
                    "name": str,
                    "status": "pass" | "fail" | "warn",
                    "message": str
                },
                ...
            ]
        }
    """
    if not extracted_data:
        return {
            "valid": False,
            "score": 0,
            "checks": [
                {
                    "name": "Data presence",
                    "status": "fail",
                    "message": "No extracted data provided for validation"
                }
            ],
        }

    document_type = (document_type or "").strip().lower()

    if document_type == "passport":
        checks = _validate_passport(extracted_data)
    elif document_type == "visa":
        checks = _validate_visa(extracted_data)
    elif document_type in ("national_id", "driving_license", "permit"):
        checks = _validate_generic_id(extracted_data, document_type)
    else:
        logger.warning("Unknown document type '%s' — running generic validation", document_type)
        checks = _validate_generic_id(extracted_data, document_type)

    # Compute score
    score = _compute_score(checks)
    valid = score >= 50 and not any(
        c["status"] == "fail" and c["name"] in _critical_checks(document_type)
        for c in checks
    )

    return {
        "valid": valid,
        "score": score,
        "checks": checks,
    }


# ===================================================================
#  PASSPORT VALIDATION
# ===================================================================

_PASSPORT_REQUIRED_FIELDS = ["name", "document_number", "date_of_birth", "date_of_expiry"]


def _validate_passport(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run all passport-specific validation checks."""
    checks: List[Dict[str, str]] = []

    # 1. Required fields present
    checks.extend(_check_required_fields(data, _PASSPORT_REQUIRED_FIELDS))

    # 2. Document number structure
    doc_num = data.get("document_number")
    if doc_num:
        if re.match(r'^[A-Z]{0,2}\d{6,9}[A-Z0-9]{0,2}$', str(doc_num).upper()):
            checks.append(_pass("Document number format",
                                "Document number structure is consistent with passport formats"))
        else:
            checks.append(_warn("Document number format",
                                f"Document number '{doc_num}' has an unusual format"))
    else:
        checks.append(_fail("Document number format",
                            "Document number is missing"))

    # 3. DOB parseability and reasonableness
    dob = data.get("date_of_birth")
    checks.extend(_check_date_field(dob, "Date of birth", allow_future=False, check_age=True))

    # 4. Expiry parseability and status
    expiry = data.get("date_of_expiry")
    checks.extend(_check_expiry(expiry))

    # 5. Date consistency: expiry must be after DOB
    checks.extend(_check_date_order(dob, expiry, "Date of birth", "Expiry date"))

    # 6. Nationality format
    nationality = data.get("nationality")
    if nationality:
        nat_clean = str(nationality).strip().upper()
        if re.match(r'^[A-Z]{2,3}$', nat_clean):
            checks.append(_pass("Nationality format",
                                "Nationality code is a valid 2-3 letter code"))
        else:
            checks.append(_warn("Nationality format",
                                f"Nationality '{nationality}' is not a standard 2-3 letter code"))
    else:
        checks.append(_warn("Nationality format",
                            "Nationality field is missing"))

    # 7. Gender value
    gender = data.get("gender")
    if gender:
        g = str(gender).strip().upper()
        if g in ("M", "F", "X", "MALE", "FEMALE"):
            checks.append(_pass("Gender value",
                                "Gender field contains a recognized value"))
        else:
            checks.append(_warn("Gender value",
                                f"Gender value '{gender}' is not a standard code (M/F/X)"))

    # 8. MRZ format check
    raw_text = data.get("raw_text", "")
    checks.extend(_check_mrz(raw_text))

    return checks


# ===================================================================
#  VISA VALIDATION
# ===================================================================

_VISA_REQUIRED_FIELDS = ["visa_number"]


def _validate_visa(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run all visa-specific validation checks."""
    checks: List[Dict[str, str]] = []

    # 1. Visa number present
    visa_num = data.get("visa_number")
    if visa_num and str(visa_num).strip():
        checks.append(_pass("Visa number presence",
                            "Visa number is present"))
    else:
        checks.append(_fail("Visa number presence",
                            "Visa number is missing"))

    # 2. Valid from — parseable
    valid_from = data.get("valid_from")
    if valid_from:
        parsed_from = parse_date(str(valid_from))
        if parsed_from:
            checks.append(_pass("Valid-from date",
                                "Valid-from date is parseable"))
        else:
            checks.append(_warn("Valid-from date",
                                f"Valid-from date '{valid_from}' could not be parsed"))
    else:
        checks.append(_warn("Valid-from date",
                            "Valid-from date is missing"))

    # 3. Valid until — parseable
    valid_until = data.get("valid_until")
    if valid_until:
        parsed_until = parse_date(str(valid_until))
        if parsed_until:
            checks.append(_pass("Valid-until date",
                                "Valid-until date is parseable"))
            # Check if expired
            if parsed_until < date.today():
                checks.append(_warn("Visa expiry",
                                    "Visa validity period has ended"))
            else:
                checks.append(_pass("Visa expiry",
                                    "Visa is within its validity period"))
        else:
            checks.append(_warn("Valid-until date",
                                f"Valid-until date '{valid_until}' could not be parsed"))
    else:
        checks.append(_warn("Valid-until date",
                            "Valid-until date is missing"))

    # 4. Date order: valid_until > valid_from
    if valid_from and valid_until:
        parsed_from = parse_date(str(valid_from))
        parsed_until = parse_date(str(valid_until))
        if parsed_from and parsed_until:
            if parsed_until > parsed_from:
                checks.append(_pass("Date order",
                                    "Valid-until date is after valid-from date"))
            elif parsed_until == parsed_from:
                checks.append(_warn("Date order",
                                    "Valid-from and valid-until dates are the same"))
            else:
                checks.append(_fail("Date order",
                                    "Valid-until date is before valid-from date"))

    # 5. Stay duration sensible
    stay = data.get("stay_duration")
    if stay:
        duration_days = _parse_stay_duration(str(stay))
        if duration_days is not None:
            if 1 <= duration_days <= 365:
                checks.append(_pass("Stay duration",
                                    f"Stay duration ({duration_days} days) is within a reasonable range"))
            else:
                checks.append(_warn("Stay duration",
                                    f"Stay duration ({duration_days} days) seems unusual"))
        else:
            checks.append(_warn("Stay duration",
                                f"Stay duration '{stay}' could not be interpreted"))

    return checks


# ===================================================================
#  GENERIC ID VALIDATION (national_id, driving_license, permit)
# ===================================================================

_GENERIC_REQUIRED_FIELDS = ["name", "document_number"]


def _validate_generic_id(data: Dict[str, Any], document_type: str) -> List[Dict[str, str]]:
    """Run validation for national ID, driving license, or permit."""
    checks: List[Dict[str, str]] = []

    label = document_type.replace("_", " ").title()

    # 1. Required identity fields
    checks.extend(_check_required_fields(data, _GENERIC_REQUIRED_FIELDS, label=label))

    # 2. Document number present
    doc_num = data.get("document_number")
    if doc_num and str(doc_num).strip():
        checks.append(_pass("Document number",
                            f"{label} document number is present"))
    else:
        checks.append(_fail("Document number",
                            f"{label} document number is missing"))

    # 3. DOB check
    dob = data.get("date_of_birth")
    if dob:
        checks.extend(_check_date_field(dob, "Date of birth", allow_future=False, check_age=True))

    # 4. Expiry check where available
    expiry = data.get("date_of_expiry")
    if expiry:
        checks.extend(_check_expiry(expiry))

    # 5. Date consistency
    if dob and expiry:
        checks.extend(_check_date_order(dob, expiry, "Date of birth", "Expiry date"))

    return checks


# ===================================================================
#  MRZ VALIDATION
# ===================================================================

def _check_mrz(raw_text: str) -> List[Dict[str, str]]:
    """Validate MRZ format and checksums if MRZ lines are present."""
    checks: List[Dict[str, str]] = []

    if not raw_text:
        return checks

    # Look for MRZ-like lines
    mrz_pattern = re.compile(r'[A-Z0-9<]{40,50}')
    text_no_spaces = raw_text.replace(" ", "")
    matches = mrz_pattern.findall(text_no_spaces)

    mrz_lines = [m[:44] for m in matches if len(m) >= 42]

    if len(mrz_lines) < 2:
        # No MRZ detected — not a failure, just skip
        return checks

    checks.append(_pass("MRZ presence",
                        "Machine Readable Zone (MRZ) lines detected"))

    # Validate line 2 checksums
    line2 = mrz_lines[-1].ljust(44, "<")

    # Document number check digit (position 9, data 0-8)
    if len(line2) > 9 and line2[9].isdigit():
        expected = _mrz_checksum(line2[0:9])
        actual = int(line2[9])
        if expected is not None:
            if expected == actual:
                checks.append(_pass("MRZ document number checksum",
                                    "MRZ document number check digit is valid"))
            else:
                checks.append(_fail("MRZ document number checksum",
                                    f"MRZ document number check digit mismatch "
                                    f"(expected {expected}, got {actual})"))

    # DOB check digit (position 19, data 13-18)
    if len(line2) > 19 and line2[19].isdigit():
        expected = _mrz_checksum(line2[13:19])
        actual = int(line2[19])
        if expected is not None:
            if expected == actual:
                checks.append(_pass("MRZ DOB checksum",
                                    "MRZ date of birth check digit is valid"))
            else:
                checks.append(_fail("MRZ DOB checksum",
                                    f"MRZ DOB check digit mismatch "
                                    f"(expected {expected}, got {actual})"))

    # Expiry check digit (position 27, data 21-26)
    if len(line2) > 27 and line2[27].isdigit():
        expected = _mrz_checksum(line2[21:27])
        actual = int(line2[27])
        if expected is not None:
            if expected == actual:
                checks.append(_pass("MRZ expiry checksum",
                                    "MRZ expiry date check digit is valid"))
            else:
                checks.append(_fail("MRZ expiry checksum",
                                    f"MRZ expiry check digit mismatch "
                                    f"(expected {expected}, got {actual})"))

    return checks


def _mrz_checksum(data: str) -> Optional[int]:
    """Compute ICAO 9303 MRZ check digit."""
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
#  SHARED CHECK HELPERS
# ===================================================================

def _check_required_fields(
    data: Dict[str, Any],
    required: List[str],
    label: str = "Document",
) -> List[Dict[str, str]]:
    """Check that all required fields are present and non-empty."""
    checks = []
    missing = []
    for field in required:
        val = data.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(field)

    if not missing:
        checks.append(_pass("Required fields",
                            f"All required fields are present"))
    else:
        readable = ", ".join(f.replace("_", " ") for f in missing)
        checks.append(_fail("Required fields",
                            f"Missing required field(s): {readable}"))
    return checks


def _check_date_field(
    date_str: Any,
    field_name: str,
    allow_future: bool = False,
    check_age: bool = False,
) -> List[Dict[str, str]]:
    """Validate a single date field for parseability and reasonableness."""
    checks = []

    if date_str is None or (isinstance(date_str, str) and not date_str.strip()):
        checks.append(_warn(f"{field_name} presence",
                            f"{field_name} is missing"))
        return checks

    parsed = parse_date(str(date_str))
    if parsed is None:
        checks.append(_fail(f"{field_name} format",
                            f"{field_name} '{date_str}' could not be parsed as a valid date"))
        return checks

    checks.append(_pass(f"{field_name} format",
                        f"{field_name} is a valid, parseable date"))

    # Future check
    if not allow_future and parsed > date.today():
        checks.append(_fail(f"{field_name} validity",
                            f"{field_name} is in the future, which is not expected"))

    # Age reasonableness for DOB
    if check_age:
        age = date.today().year - parsed.year
        if age < 0 or age > 150:
            checks.append(_warn(f"{field_name} reasonableness",
                                f"Implied age ({age}) is outside the expected 0–150 range"))
        else:
            checks.append(_pass(f"{field_name} reasonableness",
                                f"Implied age ({age}) is within a reasonable range"))

    return checks


def _check_expiry(expiry_str: Any) -> List[Dict[str, str]]:
    """Check expiry date for parseability and whether document has expired."""
    checks = []

    if expiry_str is None or (isinstance(expiry_str, str) and not expiry_str.strip()):
        checks.append(_warn("Expiry date presence",
                            "Expiry date is missing"))
        return checks

    parsed = parse_date(str(expiry_str))
    if parsed is None:
        checks.append(_fail("Expiry date format",
                            f"Expiry date '{expiry_str}' could not be parsed"))
        return checks

    checks.append(_pass("Expiry date format",
                        "Expiry date is a valid, parseable date"))

    if parsed < date.today():
        checks.append(_warn("Expiry date",
                            "Document has expired"))
    else:
        checks.append(_pass("Expiry date",
                            "Document has not expired"))

    return checks


def _check_date_order(
    earlier_str: Any,
    later_str: Any,
    earlier_name: str,
    later_name: str,
) -> List[Dict[str, str]]:
    """Check that the earlier date is before the later date."""
    checks = []

    if earlier_str is None or later_str is None:
        return checks

    earlier = parse_date(str(earlier_str))
    later = parse_date(str(later_str))

    if earlier is None or later is None:
        return checks

    if later > earlier:
        checks.append(_pass("Date consistency",
                            f"{later_name} is after {earlier_name} — dates are consistent"))
    elif later == earlier:
        checks.append(_warn("Date consistency",
                            f"{later_name} is the same as {earlier_name}"))
    else:
        checks.append(_fail("Date consistency",
                            f"{later_name} is before {earlier_name} — dates are inconsistent"))

    return checks


# ===================================================================
#  STAY DURATION PARSER
# ===================================================================

def _parse_stay_duration(stay_str: str) -> Optional[int]:
    """Parse a stay duration string into number of days."""
    if not stay_str:
        return None

    s = stay_str.strip().upper()

    # Try extracting number + unit
    match = re.search(r'(\d+)\s*(DAYS?|D|MONTHS?|M|WEEKS?|W|YEARS?|Y)?', s)
    if not match:
        return None

    value = int(match.group(1))
    unit = (match.group(2) or "D").strip()

    if unit.startswith("D"):
        return value
    elif unit.startswith("W"):
        return value * 7
    elif unit.startswith("M"):
        return value * 30
    elif unit.startswith("Y"):
        return value * 365
    return value


# ===================================================================
#  SCORING
# ===================================================================

def _compute_score(checks: List[Dict[str, str]]) -> int:
    """Compute a validation score (0–100) based on check results."""
    if not checks:
        return 0

    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "pass")
    warned = sum(1 for c in checks if c["status"] == "warn")

    # Warnings count as half
    score = ((passed + warned * 0.5) / total) * 100
    return max(0, min(100, round(score)))


def _critical_checks(document_type: str) -> List[str]:
    """Return check names that are considered critical (blocking)."""
    if document_type == "passport":
        return ["Required fields", "Date consistency"]
    elif document_type == "visa":
        return ["Visa number presence", "Date order"]
    else:
        return ["Required fields"]


# ===================================================================
#  CHECK RESULT BUILDERS
# ===================================================================

def _pass(name: str, message: str) -> Dict[str, str]:
    return {"name": name, "status": "pass", "message": message}


def _fail(name: str, message: str) -> Dict[str, str]:
    return {"name": name, "status": "fail", "message": message}


def _warn(name: str, message: str) -> Dict[str, str]:
    return {"name": name, "status": "warn", "message": message}
