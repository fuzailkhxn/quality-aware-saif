"""
src/training/trainer.py
=========================
Custom training loop integrating all quality-aware components.

What this module does
---------------------
1. Sets up the embedding model + adaptive margin loss.
2. Runs a custom training loop (instead of model.fit) so we can:
   a. Compute quality scores per batch.
   b. Pass quality scores into the loss function.
   c. Update prototypes after each batch.
3. Periodically saves model weights and prototypes.
4. Logs metrics (loss, accuracy, val_loss, val_accuracy) to a CSV file.
5. Implements early stopping.

Why a custom loop?
------------------
Keras model.fit() does not give easy access to per-sample quality weights
inside the loss function. A custom tf.GradientTape loop gives full control
over what happens inside each training step.
"""

import os
import csv
import time
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List

from src.models.prototype_memory import PrototypeMemory
from src.preprocessing.quality_assessment import batch_quality_scores


class QualityAwareTrainer:
    """
    Trainer that integrates:
      - Adaptive Margin Loss (Objectives C, D)
      - Quality assessment per batch (Objective D)
      - Prototype memory updates (Objectives A, B)
      - Logging, checkpointing, early stopping

    Parameters
    ----------
    embedding_model : tf.keras.Model
        The embedding model (output: L2-normalised vectors, shape (B, D)).
    loss_fn : AdaptiveMarginLoss
        The adaptive margin loss instance.
    prototype_memory : PrototypeMemory
        Pre-built prototype memory with num_classes matching loss_fn.
    num_classes : int
        Number of identity classes.
    learning_rate : float
        Adam learning rate. Default 1e-4.
    results_dir : str
        Directory where logs and checkpoints are saved.
    prototype_init_batches : int
        Number of initial batches to collect for prototype initialisation
        before starting the main training loop.
    quality_blur_threshold : float
        Passed to quality assessment.
    """

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        loss_fn,
        prototype_memory: PrototypeMemory,
        num_classes: int,
        learning_rate: float = 1e-4,
        results_dir: str = "results",
        prototype_init_batches: int = 5,
        quality_blur_threshold: float = 100.0,
    ) -> None:
        self.model = embedding_model
        self.loss_fn = loss_fn
        self.prototype_memory = prototype_memory
        self.num_classes = num_classes
        self.results_dir = results_dir
        self.prototype_init_batches = prototype_init_batches
        self.quality_blur_threshold = quality_blur_threshold

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

        # Gather all trainable variables: model + loss weight matrix
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        vars_list = self.trainable_variables
        if vars_list:
            try:
                self.optimizer.build(vars_list)
            except Exception:
                pass

        self._log_path = os.path.join(results_dir, "logs", "training_log.csv")
        self._init_log_file()

        # Track best validation loss for early stopping
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """
        Return the list of actual trainable variables for gradient descent.
        Includes model trainable variables plus the loss function's weight matrix.
        Prototypes in PrototypeMemory are intentionally excluded as they are
        updated via empirical EMA, not gradient descent.
        """
        vars_list = list(self.model.trainable_variables)
        if hasattr(self.loss_fn, "W") and isinstance(self.loss_fn.W, tf.Variable):
            vars_list.append(self.loss_fn.W)
        return vars_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights_path(path: str) -> str:
        """
        Resolve a weights filepath ensuring it ends with .weights.h5 for Keras 3 compatibility.

        Parameters
        ----------
        path : str
            Filepath or template name (e.g. 'models/embedding_model.h5',
            'models/embedding_model_weights.h5', or 'models/embedding_model.weights.h5').

        Returns
        -------
        str
            A normalized filepath ending strictly in '.weights.h5'.
        """
        if path.endswith(".weights.h5"):
            return path
        if path.endswith("_weights.h5"):
            return path[:-11] + ".weights.h5"
        if path.endswith(".h5"):
            return path[:-3] + ".weights.h5"
        if path.endswith(".keras"):
            return path[:-6] + ".weights.h5"
        return path + ".weights.h5"

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 30,
        patience: int = 7,
        model_save_path: str = "models/embedding_model.weights.h5",
        prototype_save_path: str = "models/prototypes.npy",
        steps_per_epoch: Optional[int] = None,
        val_steps: Optional[int] = None,
    ) -> dict:
        """
        Run the full training loop.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Yields (images, one_hot_labels) batches.
        val_dataset : tf.data.Dataset
            Yields (images, one_hot_labels) batches for validation.
        epochs : int
            Maximum number of training epochs.
        patience : int
            Early stopping: stop if val_loss does not improve for this many epochs.
        model_save_path : str
            Where to save the best embedding model weights (.weights.h5 format).
        prototype_save_path : str
            Where to save the prototype memory.

        Returns
        -------
        dict
            History dictionary with keys: loss, val_loss, accuracy, val_accuracy.
        """
        weights_save_path = self._resolve_weights_path(model_save_path)
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        print("\n" + "=" * 60)
        print("  Quality-Aware Face Recognition Training")
        print("=" * 60)

        # Step 1 — Initialise prototypes using first few batches
        print(f"\n[Init] Collecting embeddings from {self.prototype_init_batches} "
              f"batches to initialise prototypes...")
        self._initialise_prototypes(train_dataset)
        print("[Init] Prototypes initialised.\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self._run_epoch(
                train_dataset,
                training=True, steps=steps_per_epoch
            )
            val_loss, val_acc = self._run_epoch(
                val_dataset,
                training=False, steps=val_steps
            )

            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:03d}/{epochs:03d} "
                f"| loss={train_loss:.4f} acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"| {elapsed:.1f}s"
            )

            # Log metrics
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            self._append_log(epoch, train_loss, train_acc, val_loss, val_acc)

            # Checkpoint on improvement
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                os.makedirs(os.path.dirname(os.path.abspath(weights_save_path)), exist_ok=True)
                self.model.save_weights(weights_save_path)
                self.prototype_memory.save(prototype_save_path)
                print(f"  [OK] Saved best model (val_loss={val_loss:.4f})")
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                    break

        print("\nTraining complete.")
        print(f"Best val_loss: {self._best_val_loss:.4f}")
        print(f"Model weights saved -> {weights_save_path}")
        print(f"Prototypes saved    -> {prototype_save_path}")
        print(f"Training log        -> {self._log_path}")

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise_prototypes(self, dataset: tf.data.Dataset) -> None:
        """Collect embeddings from the first N batches to initialise prototypes."""
        all_embeddings: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for i, (images, labels) in enumerate(dataset):
            if i >= self.prototype_init_batches:
                break
            embeddings = self.model(images, training=False).numpy()
            int_labels = np.argmax(labels.numpy(), axis=1)
            all_embeddings.append(embeddings)
            all_labels.append(int_labels)

        if all_embeddings:
            emb_concat = np.concatenate(all_embeddings, axis=0)
            lbl_concat = np.concatenate(all_labels, axis=0)
            self.prototype_memory.initialise_from_embeddings(emb_concat, lbl_concat)

    def _run_epoch(
        self,
        dataset: tf.data.Dataset,
        training: bool,
        steps: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Run one epoch of training or validation. Returns (mean_loss, mean_acc)."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step_count = 0

        for batch_images, batch_labels in dataset:
            if steps is not None and step_count >= steps:
                break

            # Compute quality scores on raw batch (before float normalisation)
            raw_np = (batch_images.numpy() * 255).astype(np.uint8)
            quality = batch_quality_scores(
                raw_np, blur_threshold=self.quality_blur_threshold
            )
            quality_tf = tf.constant(quality, dtype=tf.float32)
            prototypes_tf = tf.constant(self.prototype_memory.get_prototypes(), dtype=tf.float32)

            if training:
                loss_val, preds, step_embeddings = self._train_step(
                    batch_images, batch_labels, quality_tf, prototypes_tf
                )
                # Update prototypes after each training step using quality-weighted EMA
                int_labels = np.argmax(batch_labels.numpy(), axis=1)
                self.prototype_memory.update(step_embeddings.numpy(), int_labels, quality)
            else:
                loss_val, preds = self._val_step(
                    batch_images, batch_labels, quality_tf, prototypes_tf
                )

            # Accuracy
            pred_classes = tf.argmax(preds, axis=1)
            true_classes = tf.argmax(batch_labels, axis=1)
            correct = tf.reduce_sum(tf.cast(pred_classes == true_classes, tf.int32)).numpy()

            batch_size = batch_images.shape[0] or 1
            total_loss += float(loss_val) * batch_size
            total_correct += int(correct)
            total_samples += batch_size
            step_count += 1

        if total_samples == 0:
            return 0.0, 0.0

        mean_loss = total_loss / total_samples
        mean_acc = total_correct / total_samples
        return mean_loss, mean_acc

    @tf.function
    def _train_step(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
        quality: tf.Tensor,
        prototypes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        vars_list = self.trainable_variables
        with tf.GradientTape() as tape:
            embeddings = self.model(images, training=True)
            loss = self.loss_fn(
                labels, embeddings, quality_scores=quality, prototypes=prototypes
            )

        grads = tape.gradient(loss, vars_list)
        # Filter None gradients (e.g. from frozen backbone layers) and pair with variables
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_list) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

        # For accuracy computation: cosine similarity logits
        W_norm = tf.math.l2_normalize(self.loss_fn.W, axis=0)
        logits = tf.matmul(tf.math.l2_normalize(embeddings, axis=1), W_norm)
        return loss, logits, embeddings

    def train_single_step(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[float, float]:
        """
        Execute exactly one training step on a single batch.
        Useful for verification, testing, and debugging.
        """
        raw_np = (images.numpy() * 255).astype(np.uint8)
        quality = batch_quality_scores(
            raw_np, blur_threshold=self.quality_blur_threshold
        )
        quality_tf = tf.constant(quality, dtype=tf.float32)
        prototypes_tf = tf.constant(self.prototype_memory.get_prototypes(), dtype=tf.float32)

        loss_val, preds, step_embeddings = self._train_step(images, labels, quality_tf, prototypes_tf)
        int_labels = np.argmax(labels.numpy(), axis=1)
        self.prototype_memory.update(step_embeddings.numpy(), int_labels, quality)

        pred_classes = tf.argmax(preds, axis=1)
        true_classes = tf.argmax(labels, axis=1)
        correct = tf.reduce_sum(tf.cast(pred_classes == true_classes, tf.int32)).numpy()
        acc = float(correct) / float(images.shape[0])
        return float(loss_val), acc

    @tf.function
    def _val_step(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
        quality: tf.Tensor,
        prototypes: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        embeddings = self.model(images, training=False)
        loss = self.loss_fn(
            labels, embeddings, quality_scores=quality, prototypes=prototypes
        )
        W_norm = tf.math.l2_normalize(self.loss_fn.W, axis=0)
        logits = tf.matmul(tf.math.l2_normalize(embeddings, axis=1), W_norm)
        return loss, logits

    def _init_log_file(self) -> None:
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

    def _append_log(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}"])
