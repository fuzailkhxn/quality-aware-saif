"""
src/models/prototype_memory.py
================================
Adaptive Empirical Prototype Memory — Objectives A and B.

What is a prototype?
--------------------
Each identity (person) is represented by a single prototype vector:
a 512-dimensional unit-norm vector that is the "average face direction"
for that person in the embedding space.

How are prototypes updated?
---------------------------
After each training batch we apply a quality-weighted exponential moving
average (EMA):

    P_c  <-  normalise( (1 - alpha*w_i) * P_c  +  alpha*w_i * f_i )

where:
  P_c   = current prototype for class c
  f_i   = L2-normalised embedding of training sample i
  alpha = prototype momentum (configured in config.py, e.g. 0.1)
  w_i   = quality weight of sample i  in [0, 1]
         (higher for sharp, well-lit faces; lower for blurry/dark ones)

The normalisation after the update keeps the prototype on the unit
hypersphere so cosine similarity remains meaningful.

Why quality-weighted?
---------------------
If a blurry image is used to update a prototype, it would drag the prototype
away from the true face direction. Down-weighting noisy samples keeps the
prototype accurate and stable (Objective B, D).
"""

import os
import numpy as np
from typing import Dict, Optional


class PrototypeMemory:
    """
    Stores and updates per-class empirical prototypes.

    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    embedding_dim : int
        Dimensionality of each embedding vector. Default 512.
    momentum : float
        EMA momentum alpha. Typical range 0.05 - 0.2.
        Lower -> faster adaptation; Higher -> more stable.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        momentum: float = 0.1,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        if not (0.0 < momentum < 1.0):
            raise ValueError("momentum must be strictly between 0 and 1.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.momentum = momentum

        # Prototypes: shape (num_classes, embedding_dim)
        # Initialised to zero; will be set in initialise_from_embeddings()
        self._prototypes: np.ndarray = np.zeros(
            (num_classes, embedding_dim), dtype=np.float32
        )

        # Track how many updates each class has received
        self._update_counts: np.ndarray = np.zeros(num_classes, dtype=np.int32)

        # Flag: False until initialise_from_embeddings() is called
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise_from_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Initialise prototypes as the class-mean of provided embeddings.

        This should be called once at the start of training using the
        embeddings collected from the first few batches.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, embedding_dim)
            L2-normalised embeddings.
        labels : np.ndarray, shape (N,)
            Integer class labels (0 ... num_classes-1).
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                # No samples for this class yet -> leave as zero vector;
                # will be updated when samples arrive during training.
                continue
            class_embeddings = embeddings[mask]
            mean_vec = class_embeddings.mean(axis=0)
            self._prototypes[c] = self._l2_normalize_vector(mean_vec)

        self._initialised = True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        quality_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update prototypes using a batch of embeddings and their labels.

        Applies the quality-weighted EMA rule per class.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, embedding_dim)
            L2-normalised embeddings from the current batch.
        labels : np.ndarray, shape (N,)
            Integer class labels.
        quality_weights : np.ndarray, shape (N,) or None
            Per-sample quality weights in [0, 1]. If None, all weights
            are set to 1.0 (equal contribution).
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        if quality_weights is None:
            quality_weights = np.ones(len(labels), dtype=np.float32)
        else:
            quality_weights = np.clip(
                np.array(quality_weights, dtype=np.float32), 0.0, 1.0
            )

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue

            class_embs = embeddings[mask]           # (M, D)
            class_weights = quality_weights[mask]   # (M,)

            # Quality-weighted mean of this class's embeddings in the batch
            total_weight = class_weights.sum() + 1e-8
            weighted_mean = (class_embs * class_weights[:, None]).sum(axis=0) / total_weight

            # EMA update
            alpha = self.momentum * float(class_weights.mean())
            updated = (1.0 - alpha) * self._prototypes[c] + alpha * weighted_mean

            # Project back onto unit hypersphere
            self._prototypes[c] = self._l2_normalize_vector(updated)
            self._update_counts[c] += int(mask.sum())

        if self._update_counts.sum() > 0:
            self._initialised = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_prototypes(self) -> np.ndarray:
        """
        Return the prototype matrix.

        Returns
        -------
        np.ndarray, shape (num_classes, embedding_dim)
        """
        return self._prototypes.copy()

    def get_prototype(self, class_idx: int) -> np.ndarray:
        """Return the prototype vector for a single class."""
        return self._prototypes[class_idx].copy()

    def similarity_to_prototypes(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between one embedding and all prototypes.

        Parameters
        ----------
        embedding : np.ndarray, shape (embedding_dim,) or (1, embedding_dim)

        Returns
        -------
        np.ndarray, shape (num_classes,)
            Similarity score in [-1, 1] for each class.
        """
        emb = np.array(embedding, dtype=np.float32).reshape(-1)
        emb = self._l2_normalize_vector(emb)
        return self._prototypes @ emb   # (num_classes,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save prototypes and metadata to a .npy file.

        Parameters
        ----------
        path : str  e.g. "models/prototypes.npy"
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, {
            "prototypes": self._prototypes,
            "update_counts": self._update_counts,
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "momentum": self.momentum,
        })

    @classmethod
    def load(cls, path: str) -> "PrototypeMemory":
        """
        Load a previously saved PrototypeMemory.

        Parameters
        ----------
        path : str  Path to the .npy file.

        Returns
        -------
        PrototypeMemory instance with restored state.
        """
        data = np.load(path, allow_pickle=True).item()
        obj = cls(
            num_classes=int(data["num_classes"]),
            embedding_dim=int(data["embedding_dim"]),
            momentum=float(data["momentum"]),
        )
        obj._prototypes = data["prototypes"]
        obj._update_counts = data["update_counts"]
        obj._initialised = True
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def class_update_counts(self) -> Dict[int, int]:
        """Return a dict mapping class index -> number of updates received."""
        return {i: int(self._update_counts[i]) for i in range(self.num_classes)}

    def is_initialised(self) -> bool:
        """Return True if prototypes have been initialised or updated."""
        return self._initialised

    def total_updates(self) -> int:
        """Return total number of updates applied across all classes."""
        return int(self._update_counts.sum())

    def has_prototype(self, class_idx: int) -> bool:
        """Return True if class_idx has received at least one prototype update."""
        if 0 <= class_idx < self.num_classes:
            return bool(self._update_counts[class_idx] > 0)
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize_vector(v: np.ndarray) -> np.ndarray:
        """Normalise a 1-D vector to unit length."""
        norm = np.linalg.norm(v) + 1e-8
        return v / norm

    def __repr__(self) -> str:
        return (
            f"PrototypeMemory(num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"momentum={self.momentum}, "
            f"initialised={self._initialised})"
        )
