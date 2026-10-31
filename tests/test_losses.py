"""
tests/test_losses.py
=====================
Unit tests for adaptive margin loss (src/training/losses.py).

Can be executed with either:
    python -m unittest tests/test_losses.py
or:
    pytest tests/test_losses.py -v
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import tensorflow as tf
    from src.training.losses import AdaptiveMarginLoss
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed in current environment")
class TestAdaptiveMarginLoss(unittest.TestCase):

    def setUp(self):
        self.num_classes = 3
        self.embedding_dim = 32
        self.loss_fn = AdaptiveMarginLoss(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            scale=16.0,
            base_margin=0.35,
            adaptive_lambda=0.5,
            margin_min=0.1,
            margin_max=0.7,
            noise_weight_floor=0.1,
        )

    def _random_embeddings(self, batch_size=6):
        rng = np.random.RandomState(42)
        embs = rng.randn(batch_size, self.embedding_dim).astype(np.float32)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        return tf.constant(embs)

    def test_loss_output_is_scalar_and_positive(self):
        embeddings = self._random_embeddings(6)
        labels = tf.one_hot([0, 1, 2, 0, 1, 2], self.num_classes)
        loss = self.loss_fn(labels, embeddings)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss.numpy()), 0.0)

    def test_loss_with_quality_scores(self):
        embeddings = self._random_embeddings(4)
        labels = tf.one_hot([0, 1, 0, 1], self.num_classes)
        quality = tf.constant([1.0, 0.9, 0.2, 0.1], dtype=tf.float32)
        loss = self.loss_fn(labels, embeddings, quality_scores=quality)
        self.assertFalse(np.isnan(float(loss.numpy())))
        self.assertGreater(float(loss.numpy()), 0.0)

    def test_get_config(self):
        cfg = self.loss_fn.get_config()
        self.assertEqual(cfg["num_classes"], self.num_classes)
        self.assertEqual(cfg["embedding_dim"], self.embedding_dim)
        self.assertIn("base_margin", cfg)
        self.assertIn("adaptive_lambda", cfg)


if __name__ == "__main__":
    unittest.main()

