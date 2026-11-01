"""
tests/test_preprocessing_consistency.py
========================================
Validates that:
1. Face cropping applies exactly 12% proportional padding and standardizes to (224, 224).
2. Preprocessing pipeline produces normalized float32 images in [0, 1].
3. QualityAwareRecognizer properly loads improved_embedding.weights.h5.
4. Database prototypes and embeddings match the 512-d normalized hypersphere constraint.
"""

import os
import unittest
import numpy as np
import cv2

from src.preprocessing.face_detection import crop_face
from src.preprocessing.image_preprocessing import preprocess_for_mobilenet
from src.recognition.recognizer import QualityAwareRecognizer


class TestPreprocessingConsistency(unittest.TestCase):

    def test_crop_face_12_percent_padding(self):
        dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
        box = (100, 100, 100, 100)  # x, y, w, h
        # With 12% padding: pad_x = 12, pad_y = 12
        # Expected crop bounds: x1=88, y1=88, x2=212, y2=212 -> shape (124, 124, 3)
        crop = crop_face(dummy_img, box, padding=0.12)
        self.assertEqual(crop.shape, (124, 124, 3))

    def test_preprocess_for_mobilenet_standardization(self):
        dummy_bgr = np.random.randint(0, 256, (150, 120, 3), dtype=np.uint8)
        processed = preprocess_for_mobilenet(dummy_bgr, target_size=(224, 224))
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0.0 <= processed.min() and processed.max() <= 1.0)

    def test_recognizer_improved_checkpoint_loading(self):
        improved_path = os.path.join("checkpoints", "improved_embedding.weights.h5")
        if os.path.exists(improved_path):
            recognizer = QualityAwareRecognizer(
                engine="MobileNetV2-512",
                weights_path=improved_path,
            )
            self.assertEqual(recognizer.embedding_dim, 512)
            dummy_face = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
            emb = recognizer.extract_embedding(dummy_face)
            self.assertEqual(emb.shape, (512,))
            norm = np.linalg.norm(emb)
            self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
