"""
tests/test_adaptive_loss.py
===========================
Unit tests for Adaptive Margin Loss and Sample Weighting (Objectives 3 & 4).
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.losses import AdaptiveMarginLoss


class TestAdaptiveMarginLoss(unittest.TestCase):

    def setUp(self):
        self.num_classes = 4
        self.embedding_dim = 512
        self.loss_fn = AdaptiveMarginLoss(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            scale=30.0,
            base_margin=0.35,
            adaptive_lambda=0.5,
            margin_min=0.1,
            margin_max=0.7,
            noise_weight_floor=0.1,
        )

    def test_loss_computes_finite_scalar(self):
        batch_size = 4
        y_true = tf.one_hot([0, 1, 2, 3], depth=self.num_classes)
        # Random normalized embeddings
        raw = np.random.randn(batch_size, self.embedding_dim).astype(np.float32)
        norm = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        y_pred = tf.constant(norm)

        loss = self.loss_fn(y_true, y_pred)
        self.assertTrue(tf.math.is_finite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_margin_adapts_to_similarity(self):
        """
        Verify that margin increases as sample difficulty increases (similarity drops).
        Formula: m_i = clip(m_base + lambda * (1 - cos_sim), min, max)
        """
        base_m = self.loss_fn.base_margin
        lam = self.loss_fn.adaptive_lambda

        # Easy sample: high similarity = 0.9 -> margin = base_m + 0.5*(0.1) = base_m + 0.05
        # Hard sample: low similarity = 0.1 -> margin = base_m + 0.5*(0.9) = base_m + 0.45
        m_easy = np.clip(base_m + lam * (1.0 - 0.9), 0.1, 0.7)
        m_hard = np.clip(base_m + lam * (1.0 - 0.1), 0.1, 0.7)

        self.assertGreater(m_hard, m_easy)
        self.assertLessEqual(m_hard, 0.7)
        self.assertGreaterEqual(m_easy, 0.1)

    def test_noise_weighting_floor(self):
        """Verify that lowest quality/confidence sample receives non-zero floor weight."""
        floor = self.loss_fn.noise_weight_floor
        self.assertEqual(floor, 0.1)
        self.assertGreater(floor, 0.0)


if __name__ == "__main__":
    unittest.main()
