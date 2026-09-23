"""
Date parsing and validation utilities for document processing.

Supports multiple international date formats commonly found on
passports, visas, national IDs, and driving licenses.
"""

import re
from datetime import datetime, date
from typing import Optional


# Common date formats found on identity documents
DATE_FORMATS = [
    "%Y-%m-%d",       # 2025-04-12
    "%d/%m/%Y",       # 12/04/2025
    "%m/%d/%Y",       # 04/12/2025
    "%d-%m-%Y",       # 12-04-2025
    "%d.%m.%Y",       # 12.04.2025
    "%d %b %Y",       # 12 Apr 2025
    "%d %B %Y",       # 12 April 2025
    "%d-%b-%Y",       # 12-Apr-2025
    "%b %d, %Y",      # Apr 12, 2025
    "%B %d, %Y",      # April 12, 2025
    "%Y/%m/%d",       # 2025/04/12
    "%Y%m%d",         # 20250412  (MRZ format)
]

# Months for manual fallback parsing
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_date(date_string: Optional[str]) -> Optional[date]:
    """
    Attempt to parse a date string using multiple common formats.

    Args:
        date_string: Raw date string from OCR output.

    Returns:
        A date object if parsing succeeds, None otherwise.
    """
    if not date_string or not isinstance(date_string, str):
        return None

    cleaned = date_string.strip()
    if not cleaned:
        return None

    # Try MRZ 6-digit format FIRST (YYMMDD) to avoid %Y%m%d misparse
    mrz_match = re.match(r'^(\d{2})(\d{2})(\d{2})$', cleaned)
    if mrz_match:
        yy, mm, dd = int(mrz_match.group(1)), int(mrz_match.group(2)), int(mrz_match.group(3))
        # MRZ convention: years 00-49 → 2000s, 50-99 → 1900s
        year = 2000 + yy if yy < 50 else 1900 + yy
        try:
            return date(year, mm, dd)
        except ValueError:
            pass

    # Try each known format
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    # Fallback: try to extract a date with regex
    # Pattern: digits-separator-alpha/digits-separator-digits
    match = re.search(
        r'(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{2,4})',
        cleaned
    )
    if match:
        d, m, y = match.group(1), match.group(2), match.group(3)
        try:
            year = int(y)
            if year < 100:
                year += 2000 if year < 50 else 1900
            return date(year, int(m), int(d))
        except (ValueError, OverflowError):
            pass

    return None


def is_expired(date_string: Optional[str]) -> Optional[bool]:
    """
    Check if a date represents a past (expired) date.

    Returns:
        True if expired, False if not expired, None if unparseable.
    """
    parsed = parse_date(date_string)
    if parsed is None:
        return None
    return parsed < date.today()


def is_date_reasonable(date_string: Optional[str], allow_future: bool = False) -> Optional[bool]:
    """
    Check if a parsed date is within a reasonable range.
    For DOB: not in future, person age 0–150.
    For expiry: allow future dates.

    Returns:
        True if reasonable, False otherwise, None if unparseable.
    """
    parsed = parse_date(date_string)
    if parsed is None:
        return None

    today = date.today()

    if not allow_future and parsed > today:
        return False

    # Reject dates more than 150 years in the past
    if (today.year - parsed.year) > 150:
        return False

    return True


def normalize_date(date_string: Optional[str]) -> Optional[str]:
    """
    Parse a date string and return it in ISO format (YYYY-MM-DD).

    Returns:
        ISO-formatted date string, or None if unparseable.
    """
    parsed = parse_date(date_string)
    if parsed is None:
        return None
    return parsed.isoformat()
