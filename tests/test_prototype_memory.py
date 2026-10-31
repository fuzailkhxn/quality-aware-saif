"""
tests/test_prototype_memory.py
==============================
Unit tests for Adaptive Empirical Prototype Memory (Objectives 1 & 2).
"""

import os
import sys
import unittest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.prototype_memory import PrototypeMemory


class TestPrototypeMemory(unittest.TestCase):

    def setUp(self):
        self.num_classes = 4
        self.embedding_dim = 512
        self.memory = PrototypeMemory(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            momentum=0.1,
        )

    def test_initialization(self):
        self.assertFalse(self.memory.is_initialised())
        self.assertEqual(self.memory.num_classes, 4)
        self.assertEqual(self.memory.embedding_dim, 512)

    def test_initialise_from_embeddings(self):
        np.random.seed(42)
        embeddings = np.random.randn(20, self.embedding_dim).astype(np.float32)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.random.randint(0, self.num_classes, size=(20,))

        self.memory.initialise_from_embeddings(embeddings, labels)
        self.assertTrue(self.memory.is_initialised())

        prototypes = self.memory.get_prototypes()
        self.assertEqual(prototypes.shape, (self.num_classes, self.embedding_dim))

        # Check unit norm
        norms = np.linalg.norm(prototypes, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_dynamic_ema_update(self):
        np.random.seed(42)
        dummy_embs = np.random.randn(self.num_classes, self.embedding_dim).astype(np.float32)
        dummy_embs = dummy_embs / np.linalg.norm(dummy_embs, axis=1, keepdims=True)
        self.memory.initialise_from_embeddings(dummy_embs, np.arange(self.num_classes))

        old_proto = self.memory.get_prototype(0).copy()

        # Update with high quality sample
        sample_emb = np.random.randn(1, self.embedding_dim).astype(np.float32)
        sample_emb = sample_emb / np.linalg.norm(sample_emb)
        quality_weights = np.array([0.9], dtype=np.float32)

        self.memory.update_prototypes(sample_emb, np.array([0]), quality_weights)
        new_proto = self.memory.get_prototype(0)

        # Vector should have moved
        diff = np.linalg.norm(new_proto - old_proto)
        self.assertGreater(diff, 1e-4)

        # Norm must still be 1.0
        self.assertAlmostEqual(np.linalg.norm(new_proto), 1.0, places=5)

    def test_save_and_load(self):
        import tempfile
        dummy_embs = np.random.randn(self.num_classes, self.embedding_dim).astype(np.float32)
        dummy_embs = dummy_embs / np.linalg.norm(dummy_embs, axis=1, keepdims=True)
        self.memory.initialise_from_embeddings(dummy_embs, np.arange(self.num_classes))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "prototypes.npy")
            self.memory.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_mem = PrototypeMemory.load(save_path)
            self.assertTrue(loaded_mem.is_initialised())
            np.testing.assert_array_almost_equal(self.memory.get_prototypes(), loaded_mem.get_prototypes())

            loaded_instance = PrototypeMemory(self.num_classes, self.embedding_dim)
            loaded_instance.load_from_file(save_path)
            self.assertTrue(loaded_instance.is_initialised())
            np.testing.assert_array_almost_equal(self.memory.get_prototypes(), loaded_instance.get_prototypes())


if __name__ == "__main__":
    unittest.main()
