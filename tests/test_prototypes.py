"""
tests/test_prototypes.py
=========================
Unit tests for src/models/prototype_memory.py

Can be executed with either:
    python -m unittest tests/test_prototypes.py
or:
    pytest tests/test_prototypes.py -v
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.prototype_memory import PrototypeMemory


def _random_l2_embeddings(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Helper: create random L2-normalised embeddings."""
    rng = np.random.RandomState(seed)
    embs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms


class TestPrototypeMemory(unittest.TestCase):

    def test_basic_creation(self):
        pm = PrototypeMemory(num_classes=3, embedding_dim=64, momentum=0.1)
        self.assertEqual(pm.num_classes, 3)
        self.assertEqual(pm.embedding_dim, 64)
        self.assertFalse(pm.is_initialised())

    def test_invalid_num_classes(self):
        with self.assertRaises(ValueError):
            PrototypeMemory(num_classes=0)

    def test_invalid_momentum(self):
        with self.assertRaises(ValueError):
            PrototypeMemory(num_classes=2, momentum=0.0)
        with self.assertRaises(ValueError):
            PrototypeMemory(num_classes=2, momentum=1.0)

    def test_prototypes_shape(self):
        pm = PrototypeMemory(num_classes=5, embedding_dim=128)
        self.assertEqual(pm.get_prototypes().shape, (5, 128))

    def test_initialise_sets_flag_and_unit_norm(self):
        pm = PrototypeMemory(num_classes=2, embedding_dim=64)
        embs = _random_l2_embeddings(20, 64)
        labels = np.array([0] * 10 + [1] * 10)
        pm.initialise_from_embeddings(embs, labels)
        self.assertTrue(pm.is_initialised())

        protos = pm.get_prototypes()
        for i in range(2):
            norm = np.linalg.norm(protos[i])
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_prototype_direction_is_correct(self):
        """Prototype 0 should align with class 0 features."""
        pm = PrototypeMemory(num_classes=2, embedding_dim=16)
        c0 = np.zeros((5, 16), dtype=np.float32)
        c0[:, 0] = 1.0
        c1 = np.zeros((5, 16), dtype=np.float32)
        c1[:, 1] = 1.0
        embs = np.vstack([c0, c1])
        labels = np.array([0] * 5 + [1] * 5)
        pm.initialise_from_embeddings(embs, labels)
        protos = pm.get_prototypes()
        self.assertGreater(protos[0, 0], 0.9)
        self.assertGreater(protos[1, 1], 0.9)

    def test_update_changes_prototypes_and_preserves_unit_norm(self):
        pm = PrototypeMemory(num_classes=2, embedding_dim=32, momentum=0.1)
        embs = _random_l2_embeddings(10, 32)
        labels = np.array([0] * 5 + [1] * 5)
        pm.initialise_from_embeddings(embs, labels)
        protos_before = pm.get_prototypes().copy()

        new_embs = _random_l2_embeddings(6, 32, seed=99)
        pm.update(new_embs, np.array([0, 0, 0, 1, 1, 1]))
        protos_after = pm.get_prototypes()

        self.assertFalse(np.allclose(protos_before, protos_after))
        for i in range(2):
            self.assertAlmostEqual(np.linalg.norm(protos_after[i]), 1.0, places=4)

    def test_quality_weight_effect(self):
        """High quality sample updates prototype more significantly than noisy sample."""
        pm = PrototypeMemory(num_classes=1, embedding_dim=4, momentum=0.5)
        pm._prototypes[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pm._initialised = True

        new_emb = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        pm.update(new_emb, np.array([0]), quality_weights=np.array([1.0]))
        proto_high_q = pm.get_prototypes()[0].copy()

        pm._prototypes[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pm.update(new_emb, np.array([0]), quality_weights=np.array([0.01]))
        proto_low_q = pm.get_prototypes()[0].copy()

        self.assertGreater(proto_high_q[1], proto_low_q[1])

    def test_similarity_to_prototypes(self):
        pm = PrototypeMemory(num_classes=3, embedding_dim=16)
        embs = _random_l2_embeddings(15, 16)
        labels = np.array([0] * 5 + [1] * 5 + [2] * 5)
        pm.initialise_from_embeddings(embs, labels)
        query = _random_l2_embeddings(1, 16)[0]
        sims = pm.similarity_to_prototypes(query)
        self.assertEqual(sims.shape, (3,))
        self.assertTrue(np.all(sims >= -1.0001) and np.all(sims <= 1.0001))

    def test_save_load_roundtrip(self):
        pm = PrototypeMemory(num_classes=3, embedding_dim=32, momentum=0.15)
        embs = _random_l2_embeddings(30, 32)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        pm.initialise_from_embeddings(embs, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "protos.npy")
            pm.save(path)
            pm2 = PrototypeMemory.load(path)

            self.assertEqual(pm2.num_classes, pm.num_classes)
            self.assertEqual(pm2.embedding_dim, pm.embedding_dim)
            self.assertEqual(pm2.momentum, pm.momentum)
            self.assertTrue(pm2.is_initialised())
            self.assertTrue(np.allclose(pm.get_prototypes(), pm2.get_prototypes(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

