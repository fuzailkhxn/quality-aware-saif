"""
tests/test_trainer_step.py
==========================
Unit test for single training step execution in QualityAwareTrainer.

Verifies:
1. Model forward pass on GradientTape.
2. Loss computation with adaptive margins.
3. Gradient computation using actual tf.Variable instances.
4. Optimizer apply_gradients() execution (no SymbolicTensor errors).
5. PrototypeMemory update via quality-weighted EMA.
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
from src.models.prototype_memory import PrototypeMemory
from src.training.trainer import QualityAwareTrainer


class TestTrainerSingleStep(unittest.TestCase):

    def setUp(self):
        # Create a lightweight model for fast testing
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(512)(x)
        outputs = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.num_classes = 4
        self.embedding_dim = 512
        self.loss_fn = AdaptiveMarginLoss(num_classes=self.num_classes, embedding_dim=self.embedding_dim)
        self.proto_mem = PrototypeMemory(num_classes=self.num_classes, embedding_dim=self.embedding_dim)
        # Initialise prototypes with dummy embeddings
        dummy_embs = np.random.randn(self.num_classes, self.embedding_dim).astype(np.float32)
        dummy_lbls = np.arange(self.num_classes, dtype=np.int32)
        self.proto_mem.initialise_from_embeddings(dummy_embs, dummy_lbls)

        self.trainer = QualityAwareTrainer(
            embedding_model=self.model,
            loss_fn=self.loss_fn,
            prototype_memory=self.proto_mem,
            num_classes=self.num_classes,
            learning_rate=1e-4,
            results_dir="results",
        )

    def test_trainable_variables_are_tf_variables(self):
        """All items in trainable_variables must be actual trainable Keras/TF variables."""
        vars_list = self.trainer.trainable_variables
        self.assertGreater(len(vars_list), 0)
        for var in vars_list:
            # Must possess _unique_id required by Keras optimizers and be a variable
            self.assertTrue(
                hasattr(var, "_unique_id") or hasattr(var, "handle"),
                f"Variable {var} is missing _unique_id attribute!"
            )

    def test_single_train_step_runs_successfully(self):
        """Execute one training step and verify loss and accuracy."""
        batch_size = 4
        images = tf.constant(np.random.rand(batch_size, 224, 224, 3), dtype=tf.float32)
        # One-hot labels
        labels = tf.one_hot([0, 1, 2, 3], depth=self.num_classes)

        loss, acc = self.trainer.train_single_step(images, labels)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertFalse(np.isnan(loss))

        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        # Verify prototype memory was updated
        self.assertTrue(self.proto_mem.is_initialised())
        prototypes = self.proto_mem.get_prototypes()
        self.assertEqual(prototypes.shape, (self.num_classes, self.embedding_dim))

    def test_prototype_values_change_after_training_step(self):
        """Confirm that prototype values actually change after one training step."""
        protos_before = self.proto_mem.get_prototypes().copy()
        updates_before = self.proto_mem.total_updates()

        images = tf.constant(np.random.rand(4, 224, 224, 3), dtype=tf.float32)
        labels = tf.one_hot([0, 1, 2, 3], depth=self.num_classes)

        loss, acc = self.trainer.train_single_step(images, labels)

        protos_after = self.proto_mem.get_prototypes().copy()
        updates_after = self.proto_mem.total_updates()

        self.assertGreater(updates_after, updates_before)
        delta = float(np.linalg.norm(protos_after - protos_before))
        self.assertGreater(delta, 0.0)

        # Check unit norm of updated prototypes
        for i in range(self.num_classes):
            norm = float(np.linalg.norm(protos_after[i]))
            self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
