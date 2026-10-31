"""
tests/test_preprocessing.py
===========================
Unit tests for face detection, image preprocessing, and quality assessment.
"""

import os
import sys
import unittest
import numpy as np
import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preprocessing.face_detection import FaceDetector, crop_face
from src.preprocessing.image_preprocessing import (
    preprocess_for_mobilenet, apply_blur, apply_brightness, apply_contrast, apply_noise
)
from src.preprocessing.quality_assessment import (
    compute_sharpness, compute_brightness_score, compute_quality_score, FaceQualityAssessor
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 200x200 BGR test image with simple pattern
        self.test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(self.test_img, (100, 100), 50, (200, 180, 160), -1)  # Face circle
        cv2.circle(self.test_img, (80, 85), 8, (20, 20, 20), -1)        # Left eye
        cv2.circle(self.test_img, (120, 85), 8, (20, 20, 20), -1)       # Right eye

    def test_mobilenet_preprocessing(self):
        processed = preprocess_for_mobilenet(self.test_img, target_size=(224, 224))
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)

    def test_face_detector_fallback(self):
        detector = FaceDetector()
        faces = detector.detect_faces(self.test_img)
        self.assertGreater(len(faces), 0)
        box = faces[0]["box"]
        self.assertEqual(len(box), 4)

    def test_crop_face(self):
        cropped = crop_face(self.test_img, (50, 50, 100, 100), padding=0.1)
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)

    def test_quality_assessor(self):
        assessor = FaceQualityAssessor()
        res = assessor.assess(self.test_img)
        self.assertIn("overall_score", res)
        self.assertIn("quality_level", res)
        self.assertIn(res["quality_level"], ["High", "Medium", "Low"])
        self.assertGreaterEqual(res["overall_score"], 0.0)
        self.assertLessEqual(res["overall_score"], 100.0)

    def test_quality_degradations(self):
        blurred = apply_blur(self.test_img, ksize=21)
        sharp_original = compute_sharpness(self.test_img)
        sharp_blurred = compute_sharpness(blurred)
        # Blurred image must have lower sharpness
        self.assertLessEqual(sharp_blurred, sharp_original + 1e-4)

        darkened = apply_brightness(self.test_img, factor=0.2)
        self.assertLess(darkened.mean(), self.test_img.mean())


if __name__ == "__main__":
    unittest.main()
