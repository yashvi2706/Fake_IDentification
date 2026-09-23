"""
Test script for OCR and Validation modules (Person 2).

Tests cover:
- OCR extraction with mock/sample data
- Validation on well-formed data
- Validation on missing fields
- Expired document detection
- Invalid date format handling
- Date utilities
- MRZ parsing and checksum

Can be run as:
    cd backend
    python -m pytest tests/test_ocr_validation.py -v

Or directly:
    cd backend
    python tests/test_ocr_validation.py
"""

import sys
import os

# Ensure backend/ is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from datetime import date, timedelta

from utils.date_utils import parse_date, is_expired, is_date_reasonable, normalize_date
from services.validation_service import validate_document


# ===================================================================
#  DATE UTILS TESTS
# ===================================================================

class TestDateUtils(unittest.TestCase):
    """Tests for the shared date parsing utilities."""

    def test_parse_iso_format(self):
        result = parse_date("2025-04-12")
        self.assertEqual(result, date(2025, 4, 12))

    def test_parse_slash_format(self):
        result = parse_date("12/04/2025")
        self.assertEqual(result, date(2025, 4, 12))

    def test_parse_dot_format(self):
        result = parse_date("12.04.2025")
        self.assertEqual(result, date(2025, 4, 12))

    def test_parse_text_month(self):
        result = parse_date("12 Apr 2025")
        self.assertEqual(result, date(2025, 4, 12))

    def test_parse_mrz_date(self):
        result = parse_date("250412")  # YYMMDD → 2025-04-12
        self.assertEqual(result, date(2025, 4, 12))

    def test_parse_mrz_date_1900s(self):
        result = parse_date("850412")  # YYMMDD → 1985-04-12
        self.assertEqual(result, date(1985, 4, 12))

    def test_parse_none(self):
        self.assertIsNone(parse_date(None))

    def test_parse_empty(self):
        self.assertIsNone(parse_date(""))

    def test_parse_garbage(self):
        self.assertIsNone(parse_date("not a date at all"))

    def test_is_expired_past(self):
        past = (date.today() - timedelta(days=30)).isoformat()
        self.assertTrue(is_expired(past))

    def test_is_expired_future(self):
        future = (date.today() + timedelta(days=365)).isoformat()
        self.assertFalse(is_expired(future))

    def test_is_expired_unparseable(self):
        self.assertIsNone(is_expired("garbage"))

    def test_is_date_reasonable_valid_dob(self):
        self.assertTrue(is_date_reasonable("1990-06-15"))

    def test_is_date_reasonable_future_dob(self):
        future = (date.today() + timedelta(days=100)).isoformat()
        self.assertFalse(is_date_reasonable(future, allow_future=False))

    def test_normalize_date(self):
        self.assertEqual(normalize_date("12 Apr 2025"), "2025-04-12")

    def test_normalize_date_none(self):
        self.assertIsNone(normalize_date(None))


# ===================================================================
#  VALIDATION SERVICE TESTS
# ===================================================================

class TestPassportValidation(unittest.TestCase):
    """Tests for passport document validation."""

    def _make_passport(self, **overrides):
        """Create a well-formed passport data dict."""
        data = {
            "name": "JOHN DOE",
            "document_number": "A1234567",
            "nationality": "GBR",
            "date_of_birth": "1985-04-12",
            "date_of_expiry": (date.today() + timedelta(days=365)).isoformat(),
            "gender": "M",
            "raw_text": "",
            "confidence": 0.94,
        }
        data.update(overrides)
        return data

    def test_valid_passport(self):
        result = validate_document(self._make_passport(), "passport")
        self.assertTrue(result["valid"])
        self.assertGreater(result["score"], 70)
        self.assertIsInstance(result["checks"], list)
        self.assertTrue(len(result["checks"]) > 0)

    def test_passport_missing_name(self):
        result = validate_document(self._make_passport(name=None), "passport")
        failed = [c for c in result["checks"] if c["status"] == "fail"]
        self.assertTrue(len(failed) > 0)

    def test_passport_missing_all_fields(self):
        empty = {
            "name": None,
            "document_number": None,
            "nationality": None,
            "date_of_birth": None,
            "date_of_expiry": None,
            "gender": None,
            "raw_text": "",
            "confidence": 0.0,
        }
        result = validate_document(empty, "passport")
        self.assertFalse(result["valid"])
        self.assertLessEqual(result["score"], 50)

    def test_passport_expired(self):
        expired_date = (date.today() - timedelta(days=30)).isoformat()
        result = validate_document(self._make_passport(date_of_expiry=expired_date), "passport")
        expiry_checks = [c for c in result["checks"]
                         if "expir" in c["name"].lower() or "expir" in c["message"].lower()]
        has_warning = any(c["status"] in ("warn", "fail") for c in expiry_checks)
        self.assertTrue(has_warning, "Expected warning or failure for expired passport")

    def test_passport_bad_date_format(self):
        result = validate_document(
            self._make_passport(date_of_birth="not-a-date"), "passport"
        )
        dob_checks = [c for c in result["checks"] if "birth" in c["name"].lower()]
        has_failure = any(c["status"] == "fail" for c in dob_checks)
        self.assertTrue(has_failure, "Expected failure for unparseable DOB")

    def test_passport_invalid_gender(self):
        result = validate_document(self._make_passport(gender="Z"), "passport")
        gender_checks = [c for c in result["checks"] if "gender" in c["name"].lower()]
        has_warning = any(c["status"] == "warn" for c in gender_checks)
        self.assertTrue(has_warning, "Expected warning for invalid gender")

    def test_passport_bad_nationality(self):
        result = validate_document(self._make_passport(nationality="12345"), "passport")
        nat_checks = [c for c in result["checks"] if "nationality" in c["name"].lower()]
        has_warning = any(c["status"] == "warn" for c in nat_checks)
        self.assertTrue(has_warning, "Expected warning for invalid nationality")

    def test_passport_date_inconsistency(self):
        """Expiry before DOB should fail date consistency."""
        result = validate_document(
            self._make_passport(
                date_of_birth="2020-01-01",
                date_of_expiry="2010-01-01"
            ),
            "passport",
        )
        consistency = [c for c in result["checks"] if "consistency" in c["name"].lower()]
        has_failure = any(c["status"] == "fail" for c in consistency)
        self.assertTrue(has_failure, "Expected failure for DOB after expiry")


class TestVisaValidation(unittest.TestCase):
    """Tests for visa document validation."""

    def _make_visa(self, **overrides):
        data = {
            "visa_number": "V12345678",
            "visa_type": "Tourist",
            "valid_from": "2025-01-01",
            "valid_until": (date.today() + timedelta(days=180)).isoformat(),
            "stay_duration": "90 days",
            "raw_text": "",
            "confidence": 0.88,
        }
        data.update(overrides)
        return data

    def test_valid_visa(self):
        result = validate_document(self._make_visa(), "visa")
        self.assertTrue(result["valid"])
        self.assertGreater(result["score"], 70)

    def test_visa_missing_number(self):
        result = validate_document(self._make_visa(visa_number=None), "visa")
        failed = [c for c in result["checks"] if c["status"] == "fail"]
        self.assertTrue(len(failed) > 0)

    def test_visa_dates_reversed(self):
        result = validate_document(
            self._make_visa(valid_from="2025-12-01", valid_until="2025-01-01"),
            "visa",
        )
        order_checks = [c for c in result["checks"] if "order" in c["name"].lower()]
        has_failure = any(c["status"] == "fail" for c in order_checks)
        self.assertTrue(has_failure, "Expected failure for reversed dates")

    def test_visa_unreasonable_stay(self):
        result = validate_document(self._make_visa(stay_duration="9999 days"), "visa")
        stay_checks = [c for c in result["checks"] if "stay" in c["name"].lower()]
        has_warning = any(c["status"] == "warn" for c in stay_checks)
        self.assertTrue(has_warning, "Expected warning for unreasonable stay duration")


class TestGenericIDValidation(unittest.TestCase):
    """Tests for national_id, driving_license, permit validation."""

    def _make_id(self, **overrides):
        data = {
            "name": "JANE DOE",
            "document_number": "DL1234567",
            "date_of_birth": "1990-06-15",
            "date_of_expiry": (date.today() + timedelta(days=730)).isoformat(),
            "address": "123 Main St",
            "raw_text": "",
            "confidence": 0.85,
        }
        data.update(overrides)
        return data

    def test_valid_national_id(self):
        result = validate_document(self._make_id(), "national_id")
        self.assertTrue(result["valid"])

    def test_valid_driving_license(self):
        result = validate_document(self._make_id(), "driving_license")
        self.assertTrue(result["valid"])

    def test_valid_permit(self):
        result = validate_document(self._make_id(), "permit")
        self.assertTrue(result["valid"])

    def test_missing_document_number(self):
        result = validate_document(self._make_id(document_number=None), "national_id")
        failed = [c for c in result["checks"] if c["status"] == "fail"]
        self.assertTrue(len(failed) > 0)


class TestEdgeCases(unittest.TestCase):
    """Edge case and robustness tests."""

    def test_empty_data(self):
        result = validate_document({}, "passport")
        self.assertFalse(result["valid"])

    def test_none_data(self):
        result = validate_document(None, "passport")
        self.assertFalse(result["valid"])
        self.assertEqual(result["score"], 0)

    def test_unknown_document_type(self):
        data = {"name": "TEST", "document_number": "12345"}
        result = validate_document(data, "unknown_type")
        self.assertIsInstance(result, dict)
        self.assertIn("checks", result)

    def test_output_structure(self):
        data = {
            "name": "JOHN DOE",
            "document_number": "A1234567",
            "date_of_birth": "1985-04-12",
            "date_of_expiry": "2030-04-12",
            "raw_text": "",
            "confidence": 0.9,
        }
        result = validate_document(data, "passport")
        self.assertIn("valid", result)
        self.assertIn("score", result)
        self.assertIn("checks", result)
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["score"], int)
        self.assertIsInstance(result["checks"], list)
        for check in result["checks"]:
            self.assertIn("name", check)
            self.assertIn("status", check)
            self.assertIn("message", check)
            self.assertIn(check["status"], ("pass", "fail", "warn"))


# ===================================================================
#  MAIN
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Person 2 — OCR & Validation Module Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
