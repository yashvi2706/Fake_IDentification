import io
import os
import sys
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tampering_service
from main import app
from fastapi.testclient import TestClient


def _make_image_bytes(color=(120, 120, 120), size=(320, 220)):
    img = Image.new("RGB", size, color)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=92)
    return b.getvalue()


class TamperServiceTests(unittest.TestCase):
    def test_invalid_image_returns_safe_result(self):
        out = tampering_service.analyze_tampering(b"not-an-image")
        self.assertEqual(out["score"], 0)
        self.assertIn("details", out)
        self.assertTrue(out["details"]["no_face_detected"])

    @mock.patch("tampering_service._run_trufor", return_value=(0.73, False))
    @mock.patch("tampering_service._run_model", return_value={"photo_replacement": 0.65, "document_tamper": 0.22})
    @mock.patch("tampering_service._detect_faces", return_value=([], "ok"))
    def test_no_face_behavior(self, _d, _m, _t):
        out = tampering_service.analyze_tampering(_make_image_bytes())
        details = out["details"]
        self.assertEqual(details["face_count"], 0)
        self.assertTrue(details["no_face_detected"])
        self.assertFalse(details["photo_available"])
        self.assertEqual(details["photo_replacement"], 0)
        self.assertEqual(details["photo_probability"], 0)

    @mock.patch("tampering_service._run_trufor", return_value=(0.35, False))
    @mock.patch("tampering_service._detect_faces", return_value=([(10, 10, 90, 90), (110, 20, 190, 100)], "ok"))
    def test_face_detected_aggregates_photo_score(self, _d, _t):
        calls = []

        def fake_run_model(_rgb, _ela):
            # first call is full image doc path; next calls are per-face
            calls.append(1)
            if len(calls) == 1:
                return {"photo_replacement": 0.05, "document_tamper": 0.4}
            if len(calls) == 2:
                return {"photo_replacement": 0.2, "document_tamper": 0.0}
            return {"photo_replacement": 0.8, "document_tamper": 0.0}

        with mock.patch("tampering_service._run_model", side_effect=fake_run_model):
            out = tampering_service.analyze_tampering(_make_image_bytes())

        details = out["details"]
        self.assertEqual(details["face_count"], 2)
        self.assertFalse(details["no_face_detected"])
        self.assertTrue(details["photo_available"])
        self.assertGreaterEqual(details["photo_replacement"], 75)

    @mock.patch("tampering_service._run_trufor", return_value=(None, False))
    @mock.patch("tampering_service._detect_faces", return_value=([], "ok"))
    @mock.patch("tampering_service._run_model", return_value={"photo_replacement": 0.0, "document_tamper": 0.0})
    def test_deterministic_integer_outputs(self, *_):
        out1 = tampering_service.analyze_tampering(_make_image_bytes())
        out2 = tampering_service.analyze_tampering(_make_image_bytes())
        self.assertEqual(out1["score"], out2["score"])
        self.assertIsInstance(out1["score"], int)
        self.assertIsInstance(out1["details"]["text_manipulation"], int)
        self.assertIsInstance(out1["details"]["document_tamper"], int)

    def test_legacy_singlehead_compatibility_path(self):
        class FakeLegacy:
            def __call__(self, _x):
                import torch

                return torch.tensor([[0.1, 2.0]], dtype=torch.float32)

        old_model = tampering_service._model
        old_type = tampering_service._model_type
        try:
            tampering_service._model = FakeLegacy()
            tampering_service._model_type = "legacy_singlehead"
            rgb = Image.new("RGB", (224, 224), (0, 0, 0))
            ela = Image.new("RGB", (224, 224), (0, 0, 0))
            out = tampering_service._run_model(rgb, ela)
            self.assertGreater(out["photo_replacement"], 0.5)
            self.assertGreater(out["document_tamper"], 0.3)
        finally:
            tampering_service._model = old_model
            tampering_service._model_type = old_type


class TamperEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @mock.patch("main.analyze_tampering")
    def test_detect_schema_has_new_fields(self, mocked):
        mocked.return_value = {
            "score": 41,
            "suspicious": False,
            "details": {
                "photo_replacement": 0,
                "document_tamper": 35,
                "text_manipulation": 62,
                "metadata_anomaly": 4,
                "compression_anomaly": 5,
                "photo_probability": 0,
                "document_probability": 35,
                "trufor_probability": 62,
                "face_count": 0,
                "no_face_detected": True,
                "photo_available": False,
                "trufor_available": True,
                "trufor_downscaled": False,
                "model_status": {"tamper_model_loaded": True},
                "face_detector_status": "ok",
                "visualization": None,
            },
            "indicators": [],
        }

        files = {"file": ("id.jpg", _make_image_bytes(), "image/jpeg")}
        resp = self.client.post("/detect", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("score", data)
        self.assertIn("details", data)
        self.assertIn("photo_available", data["details"])
        self.assertIn("document_probability", data["details"])


if __name__ == "__main__":
    unittest.main()
