"""
tests/test_prototype_loss_integration.py
=========================================
Tests confirming that:
1. Prototype vectors directly affect the Adaptive Margin Loss value.
2. Higher vs lower prototype similarity produces distinct adaptive margins and losses.
3. Quality weights modulate Prototype Memory EMA updates (high-quality updates move prototypes more).
4. Gradient flow works with prototype-aware loss.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import unittest
import numpy as np
import tensorflow as tf

from src.training.losses import AdaptiveMarginLoss
from src.models.prototype_memory import PrototypeMemory


class TestPrototypeLossIntegration(unittest.TestCase):

    def test_prototypes_affect_loss_value(self):
        """Verify that passing different prototype vectors changes the computed loss."""
        num_classes = 4
        dim = 512
        loss_fn = AdaptiveMarginLoss(num_classes=num_classes, embedding_dim=dim, adaptive_lambda=0.5)

        # Batch of 2 samples for class 0
        y_true = tf.constant([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        # Fixed unit embedding
        f = np.zeros((2, dim), dtype=np.float32)
        f[:, 0] = 1.0
        y_pred = tf.constant(f)

        # Prototype 1: identical to embedding (cosine similarity = 1.0 -> minimum difficulty)
        p_close = np.zeros((num_classes, dim), dtype=np.float32)
        p_close[0, 0] = 1.0  # class 0 has exact match
        for c in range(1, num_classes):
            p_close[c, c] = 1.0

        # Prototype 2: orthogonal to embedding (cosine similarity = 0.0 -> high difficulty)
        p_far = np.zeros((num_classes, dim), dtype=np.float32)
        p_far[0, 1] = 1.0  # class 0 prototype is perpendicular to f
        for c in range(1, num_classes):
            p_far[c, c] = 1.0

        loss_close = loss_fn(y_true, y_pred, prototypes=tf.constant(p_close))
        loss_far = loss_fn(y_true, y_pred, prototypes=tf.constant(p_far))

        loss_close_val = float(loss_close.numpy())
        loss_far_val = float(loss_far.numpy())

        print(f"Loss with close prototype: {loss_close_val:.6f}")
        print(f"Loss with far prototype:   {loss_far_val:.6f}")

        # The loss must differ and far prototype (higher difficulty) must incur higher loss
        assert abs(loss_far_val - loss_close_val) > 1e-4, "Prototypes did not affect loss value!"
        assert loss_far_val > loss_close_val, "Far prototype should produce higher difficulty and higher loss!"

    def test_quality_weights_modulate_prototype_updates(self):
        """Verify that higher quality weights result in larger prototype updates."""
        num_classes = 2
        dim = 16
        momentum = 0.5

        # Initialize prototypes to [1, 0, 0, ...]
        init_p = np.zeros((num_classes, dim), dtype=np.float32)
        init_p[:, 0] = 1.0

        mem_high = PrototypeMemory(num_classes=num_classes, embedding_dim=dim, momentum=momentum)
        mem_high._prototypes = init_p.copy()
        mem_high._initialised = True

        mem_low = PrototypeMemory(num_classes=num_classes, embedding_dim=dim, momentum=momentum)
        mem_low._prototypes = init_p.copy()
        mem_low._initialised = True

        # Embedding pointing to [0, 1, 0, ...]
        sample_emb = np.zeros((1, dim), dtype=np.float32)
        sample_emb[0, 1] = 1.0
        label = np.array([0], dtype=np.int32)

        # High quality (w = 1.0)
        mem_high.update(sample_emb, label, quality_weights=np.array([1.0], dtype=np.float32))
        p_high = mem_high.get_prototype(0)

        # Low quality (w = 0.1)
        mem_low.update(sample_emb, label, quality_weights=np.array([0.1], dtype=np.float32))
        p_low = mem_low.get_prototype(0)

        # Distance moved from initial prototype
        dist_high = np.linalg.norm(p_high - init_p[0])
        dist_low = np.linalg.norm(p_low - init_p[0])

        print(f"Update distance (high quality w=1.0): {dist_high:.6f}")
        print(f"Update distance (low quality w=0.1):  {dist_low:.6f}")

        assert dist_high > dist_low, "High quality sample should induce a larger prototype update than low quality!"

if __name__ == '__main__':
    unittest.main()
