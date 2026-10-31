"""
src/utils/metrics.py
====================
Metric helper functions shared across training and evaluation modules.

All functions operate on plain NumPy arrays to remain framework-agnostic.
"""

import numpy as np
from typing import Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of two embedding matrices.

    Parameters
    ----------
    a : np.ndarray, shape (N, D)
    b : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N,)
        Element-wise cosine similarity in [-1, 1].
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=1)


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full N×N cosine similarity matrix for a set of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return normed @ normed.T


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from ROC curve arrays.

    EER is the point where False Positive Rate equals False Negative Rate
    (i.e. 1 - TPR). Lower EER is better.

    Parameters
    ----------
    fpr : np.ndarray  — False Positive Rate values
    tpr : np.ndarray  — True Positive Rate values

    Returns
    -------
    (eer, eer_threshold_index)
    """
    fnr = 1.0 - tpr
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), int(idx)


def compute_tar_at_far(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_far: float,
) -> float:
    """
    Compute True Accept Rate (TAR) at a specified False Accept Rate (FAR).

    Parameters
    ----------
    fpr         : np.ndarray — False Positive Rate array from ROC curve
    tpr         : np.ndarray — True Positive Rate array from ROC curve
    target_far  : float      — Desired FAR level (e.g. 0.001 for FAR=0.1%)

    Returns
    -------
    float : TAR value at the closest FAR point, or 0.0 if not reachable.
    """
    # Find the index where FPR is closest to (but does not exceed) target_far
    valid = fpr <= target_far
    if not np.any(valid):
        return 0.0
    idx = np.where(valid)[0][-1]
    return float(tpr[idx])


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row of a matrix so it has unit length.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N, D)  — unit-norm rows
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)
