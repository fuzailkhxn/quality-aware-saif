"""
tests/test_checkpoint_saving.py
===============================
Unit tests for model weights saving, path normalization, and reloading.
Verifies Keras 3 compatibility for '.weights.h5' checkpoints.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import tensorflow as tf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.trainer import QualityAwareTrainer
from src.training.losses import AdaptiveMarginLoss
from src.models.prototype_memory import PrototypeMemory
from recognize import load_embedding_model_for_inference


class TestWeightsPathResolution(unittest.TestCase):
    """Verify that _resolve_weights_path always normalizes to .weights.h5."""

    def test_legacy_h5_path(self):
        resolved = QualityAwareTrainer._resolve_weights_path("models/embedding_model.h5")
        self.assertEqual(resolved, "models/embedding_model.weights.h5")

    def test_underscore_weights_h5_path(self):
        resolved = QualityAwareTrainer._resolve_weights_path("models/embedding_model_weights.h5")
        self.assertEqual(resolved, "models/embedding_model.weights.h5")

    def test_already_valid_weights_h5_path(self):
        resolved = QualityAwareTrainer._resolve_weights_path("models/embedding_model.weights.h5")
        self.assertEqual(resolved, "models/embedding_model.weights.h5")

    def test_keras_path(self):
        resolved = QualityAwareTrainer._resolve_weights_path("models/embedding_model.keras")
        self.assertEqual(resolved, "models/embedding_model.weights.h5")

    def test_extensionless_path(self):
        resolved = QualityAwareTrainer._resolve_weights_path("custom/checkpoint")
        self.assertEqual(resolved, "custom/checkpoint.weights.h5")


class TestModelWeightsSaveAndReload(unittest.TestCase):
    """Verify saving and reloading weights on Keras 3 models."""

    def setUp(self):
        # Create a lightweight deterministic model for testing
        inputs = tf.keras.Input(shape=(16,))
        x = tf.keras.layers.Dense(32, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(8, activation="linear")(x)
        self.model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Separate instance with the exact same architecture
        self.model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

    def test_save_and_reload_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            sample_input = np.random.randn(4, 16).astype(np.float32)

            pred1 = self.model1(sample_input).numpy()

            # Save weights to .weights.h5
            self.model1.save_weights(weights_path)
            self.assertTrue(os.path.exists(weights_path))
            self.assertGreater(os.path.getsize(weights_path), 0)

            # Reload into a fresh model
            inputs_fresh = tf.keras.Input(shape=(16,))
            x_fresh = tf.keras.layers.Dense(32, activation="relu")(inputs_fresh)
            outputs_fresh = tf.keras.layers.Dense(8, activation="linear")(x_fresh)
            model_fresh = tf.keras.Model(inputs=inputs_fresh, outputs=outputs_fresh)

            model_fresh.load_weights(weights_path)
            pred_fresh = model_fresh(sample_input).numpy()

            np.testing.assert_allclose(pred1, pred_fresh, rtol=1e-5, atol=1e-5)


class TestTrainerCheckpointSaving(unittest.TestCase):
    """Verify that QualityAwareTrainer checkpoints save .weights.h5 without errors."""

    def setUp(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(128)(x)
        outputs = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.num_classes = 3
        self.embedding_dim = 128
        self.loss_fn = AdaptiveMarginLoss(num_classes=self.num_classes, embedding_dim=self.embedding_dim)
        self.proto_mem = PrototypeMemory(num_classes=self.num_classes, embedding_dim=self.embedding_dim)
        dummy_embs = np.random.randn(self.num_classes, self.embedding_dim).astype(np.float32)
        dummy_lbls = np.arange(self.num_classes, dtype=np.int32)
        self.proto_mem.initialise_from_embeddings(dummy_embs, dummy_lbls)

    def test_checkpoint_saving_during_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = QualityAwareTrainer(
                embedding_model=self.model,
                loss_fn=self.loss_fn,
                prototype_memory=self.proto_mem,
                num_classes=self.num_classes,
                learning_rate=1e-4,
                results_dir=tmpdir,
                prototype_init_batches=1,
            )

            # Create dummy dataset of 2 batches
            batch_x = np.random.rand(6, 224, 224, 3).astype(np.float32)
            batch_y = tf.one_hot([0, 1, 2, 0, 1, 2], depth=self.num_classes).numpy()

            train_ds = tf.data.Dataset.from_tensor_slices((batch_x, batch_y)).batch(3)
            val_ds = tf.data.Dataset.from_tensor_slices((batch_x, batch_y)).batch(3)

            save_weights_path = os.path.join(tmpdir, "test_embedding.h5")
            save_proto_path = os.path.join(tmpdir, "test_prototypes.npy")

            # Run 1 epoch
            history = trainer.train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                model_save_path=save_weights_path,
                prototype_save_path=save_proto_path,
            )

            expected_weights_file = os.path.join(tmpdir, "test_embedding.weights.h5")
            self.assertTrue(
                os.path.exists(expected_weights_file),
                f"Weights checkpoint file {expected_weights_file} was not created!"
            )
            self.assertGreater(os.path.getsize(expected_weights_file), 0)

            self.assertTrue(
                os.path.exists(save_proto_path),
                f"Prototypes file {save_proto_path} was not created!"
            )

            # Verify prototypes can be reloaded
            loaded_pm = PrototypeMemory.load(save_proto_path)
            self.assertEqual(loaded_pm.num_classes, self.num_classes)
            self.assertEqual(loaded_pm.embedding_dim, self.embedding_dim)

            # Verify history and logs
            self.assertEqual(len(history["loss"]), 1)
            self.assertEqual(len(history["val_loss"]), 1)
            log_csv = os.path.join(tmpdir, "logs", "training_log.csv")
            self.assertTrue(os.path.exists(log_csv))


if __name__ == "__main__":
    unittest.main(verbosity=2)

