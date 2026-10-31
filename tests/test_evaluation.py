"""
tests/test_evaluation.py
=========================
Unit tests for evaluation metrics and verification utilities.

Can be executed with either:
    python -m unittest tests/test_evaluation.py
or:
    pytest tests/test_evaluation.py -v
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import (
    cosine_similarity,
    pairwise_cosine_similarity,
    compute_eer,
    compute_tar_at_far,
    l2_normalize,
)

try:
    from src.evaluation.evaluate import generate_pairs, evaluate_classification_accuracy
    EVAL_MODULE_AVAILABLE = True
except ImportError:
    EVAL_MODULE_AVAILABLE = False


class TestMetrics(unittest.TestCase):

    def test_cosine_similarity_same_vector(self):
        a = np.random.randn(5, 64).astype(np.float32)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        sims = cosine_similarity(a, a)
        self.assertTrue(np.allclose(sims, 1.0, atol=1e-5))

    def test_cosine_similarity_orthogonal(self):
        a = np.zeros((1, 4), dtype=np.float32)
        a[0, 0] = 1.0
        b = np.zeros((1, 4), dtype=np.float32)
        b[0, 1] = 1.0
        sim = cosine_similarity(a, b)
        self.assertAlmostEqual(float(sim[0]), 0.0, places=5)

    def test_pairwise_cosine_similarity(self):
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)
        sim_mat = pairwise_cosine_similarity(embeddings)
        self.assertEqual(sim_mat.shape, (3, 3))
        self.assertAlmostEqual(float(sim_mat[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(sim_mat[0, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(sim_mat[0, 2]), 1.0, places=4)

    def test_l2_normalize(self):
        vecs = np.random.randn(10, 32).astype(np.float32)
        normed = l2_normalize(vecs)
        norms = np.linalg.norm(normed, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_compute_eer_perfect(self):
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        eer, idx = compute_eer(fpr, tpr)
        self.assertLess(eer, 0.05)

    def test_compute_eer_random(self):
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        eer, _ = compute_eer(fpr, tpr)
        self.assertAlmostEqual(eer, 0.5, delta=0.05)

    def test_tar_at_far(self):
        fpr = np.array([0.0005, 0.001, 0.01, 0.1])
        tpr = np.array([0.80, 0.85, 0.95, 0.99])
        tar = compute_tar_at_far(fpr, tpr, target_far=0.001)
        self.assertAlmostEqual(tar, 0.85, places=2)


@unittest.skipUnless(EVAL_MODULE_AVAILABLE, "Evaluation module requires scikit-learn and matplotlib")
class TestPairGenerationAndEval(unittest.TestCase):

    def _make_embeddings(self, n_classes=3, n_per_class=6, dim=16, seed=42):
        rng = np.random.RandomState(seed)
        embs = []
        labels = []
        for c in range(n_classes):
            center = rng.randn(dim).astype(np.float32)
            center /= (np.linalg.norm(center) + 1e-8)
            for _ in range(n_per_class):
                noise = rng.randn(dim).astype(np.float32) * 0.05
                v = center + noise
                v /= (np.linalg.norm(v) + 1e-8)
                embs.append(v)
                labels.append(c)
        return np.array(embs), np.array(labels)

    def test_pair_generation(self):
        embs, labels = self._make_embeddings(3, 6)
        scores, pair_labels, _ = generate_pairs(embs, labels, max_pairs_per_class=15)
        self.assertEqual(len(scores), len(pair_labels))
        self.assertGreater((pair_labels == 1).sum(), 0)
        self.assertGreater((pair_labels == 0).sum(), 0)
        self.assertTrue(np.all(scores >= -1.0001) and np.all(scores <= 1.0001))

    def test_prototype_classification_accuracy(self):
        dim = 16
        n_classes = 3
        prototypes = np.eye(n_classes, dim).astype(np.float32)
        embeddings = prototypes.copy()
        labels = np.arange(n_classes)
        res = evaluate_classification_accuracy(embeddings, labels, prototypes)
        self.assertEqual(res["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

