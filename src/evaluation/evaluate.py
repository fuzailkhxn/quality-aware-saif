"""
src/evaluation/evaluate.py
===========================
Face verification evaluation — Objective E.

What is face verification?
---------------------------
Given two face images A and B, the system must decide:
  SAME   — A and B are the same person
  DIFFERENT — A and B are different people

This is done by computing the cosine similarity between their embeddings
and comparing it to a threshold.

Metrics reported
-----------------
1. ROC Curve     — Receiver Operating Characteristic; plots TPR vs FPR.
2. AUC           — Area Under the ROC Curve; 1.0 is perfect, 0.5 is random.
3. EER           — Equal Error Rate; where FPR = FNR. Lower is better.
4. TAR@FAR=x%    — True Accept Rate at specified False Accept Rate thresholds.
                   e.g. TAR@FAR=0.1% tells you how many genuine pairs are
                   accepted when only 0.1% of impostor pairs are accepted.

How pairs are generated
------------------------
Given embeddings and labels, we generate:
  - Genuine pairs: two embeddings from the SAME identity
  - Impostor pairs: two embeddings from DIFFERENT identities

The similarity scores and labels (1=genuine, 0=impostor) are used to
compute all metrics via sklearn's roc_curve and auc functions.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc

from src.utils.metrics import compute_eer, compute_tar_at_far


def generate_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pairs_per_class: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate genuine and impostor pairs from a set of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        L2-normalised face embeddings.
    labels : np.ndarray, shape (N,)
        Integer identity labels.
    max_pairs_per_class : int
        Maximum genuine pairs per class (avoids combinatorial explosion).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    scores : np.ndarray, shape (M,)    — cosine similarity for each pair
    pair_labels : np.ndarray, shape (M,) — 1 = genuine, 0 = impostor
    (idx_a, idx_b) : indices of the pairs (for debugging)
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)

    genuine_scores, impostor_scores = [], []
    genuine_idx, impostor_idx = [], []

    # Genuine pairs (same class)
    for c in unique_classes:
        idxs = np.where(labels == c)[0]
        if len(idxs) < 2:
            continue
        # All pairs or a random subset
        all_pairs = [(i, j) for ii, i in enumerate(idxs) for j in idxs[ii+1:]]
        if len(all_pairs) > max_pairs_per_class:
            chosen = rng.choice(len(all_pairs), max_pairs_per_class, replace=False)
            all_pairs = [all_pairs[k] for k in chosen]
        for i, j in all_pairs:
            sim = float(embeddings[i] @ embeddings[j])
            genuine_scores.append(sim)
            genuine_idx.append((i, j))

    # Impostor pairs (different classes)
    n_genuine = len(genuine_scores)
    if n_genuine == 0:
        raise ValueError(
            "No genuine pairs found. Need at least 2 images per identity."
        )

    # Sample roughly equal number of impostor pairs
    n_impostor_target = max(n_genuine, 200)
    attempts = 0
    while len(impostor_scores) < n_impostor_target and attempts < n_impostor_target * 10:
        i, j = rng.randint(0, len(labels), size=2)
        if labels[i] != labels[j]:
            sim = float(embeddings[i] @ embeddings[j])
            impostor_scores.append(sim)
            impostor_idx.append((i, j))
        attempts += 1

    scores = np.array(genuine_scores + impostor_scores, dtype=np.float32)
    pair_labels = np.array(
        [1] * len(genuine_scores) + [0] * len(impostor_scores), dtype=np.int32
    )
    idx = genuine_idx + impostor_idx

    return scores, pair_labels, idx


def evaluate_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    far_thresholds: Optional[List[float]] = None,
    max_pairs_per_class: int = 50,
    output_dir: str = "reports",
    seed: int = 42,
    save_plots: bool = True,
) -> Dict:
    """
    Full face verification evaluation pipeline.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        L2-normalised embeddings for test set.
    labels : np.ndarray, shape (N,)
        Integer identity labels.
    far_thresholds : list of float, optional
        FAR values at which to report TAR. Default [0.001, 0.01, 0.1].
    max_pairs_per_class : int
        Max genuine pairs per class. Default 50.
    output_dir : str
        Directory for saving ROC plots and JSON report.
    seed : int
        Random seed for pair generation.
    save_plots : bool
        Whether to save ROC curve plots.

    Returns
    -------
    dict with keys:
        auc, eer, tar_at_far, n_genuine_pairs, n_impostor_pairs,
        roc_plot_path (if saved), report_path
    """
    if far_thresholds is None:
        far_thresholds = [0.001, 0.01, 0.1]

    os.makedirs(output_dir, exist_ok=True)

    print("\n[Evaluation] Generating pairs...")
    scores, pair_labels, _ = generate_pairs(
        embeddings, labels,
        max_pairs_per_class=max_pairs_per_class,
        seed=seed,
    )

    n_genuine = int((pair_labels == 1).sum())
    n_impostor = int((pair_labels == 0).sum())
    print(f"[Evaluation] Genuine pairs: {n_genuine} | Impostor pairs: {n_impostor}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(pair_labels, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))

    # EER
    eer, eer_idx = compute_eer(fpr, tpr)

    # TAR at each FAR threshold
    tar_at_far = {}
    for far in far_thresholds:
        tar = compute_tar_at_far(fpr, tpr, far)
        tar_at_far[f"TAR@FAR={far:.3f}"] = float(tar)

    # Optimal threshold at EER point
    optimal_thresh = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5
    binary_preds = (scores >= optimal_thresh).astype(int)
    
    # Calculate precision, recall, f1, accuracy at optimal threshold
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    verif_acc = float(accuracy_score(pair_labels, binary_preds))
    verif_prec = float(precision_score(pair_labels, binary_preds, zero_division=0))
    verif_rec = float(recall_score(pair_labels, binary_preds, zero_division=0))
    verif_f1 = float(f1_score(pair_labels, binary_preds, zero_division=0))

    results = {
        "auc": roc_auc,
        "eer": eer,
        "optimal_threshold": optimal_thresh,
        "accuracy": verif_acc,
        "precision": verif_prec,
        "recall": verif_rec,
        "f1_score": verif_f1,
        "n_genuine_pairs": n_genuine,
        "n_impostor_pairs": n_impostor,
        "tar_at_far": tar_at_far,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    # Print results
    print("\n" + "=" * 50)
    print("  Face Verification Results")
    print("=" * 50)
    print(f"  AUC        : {roc_auc:.4f}  (1.0 = perfect)")
    print(f"  EER        : {eer:.4f}  (0.0 = perfect)")
    print(f"  Accuracy   : {verif_acc:.4f}")
    print(f"  Precision  : {verif_prec:.4f}")
    print(f"  Recall     : {verif_rec:.4f}")
    print(f"  F1-Score   : {verif_f1:.4f}")
    for key, val in tar_at_far.items():
        print(f"  {key}: {val:.4f}")
    print("=" * 50 + "\n")

    # Save ROC plot
    if save_plots:
        roc_plot_path = os.path.join(output_dir, "roc_curve.png")
        _plot_roc(fpr, tpr, roc_auc, eer, eer_idx, roc_plot_path)
        results["roc_plot_path"] = roc_plot_path
        print(f"[Evaluation] ROC curve saved -> {roc_plot_path}")

    # Save JSON report (exclude large arrays from JSON)
    report_results = {k: v for k, v in results.items() if k not in ("fpr", "tpr", "thresholds")}
    report_path = os.path.join(output_dir, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_results, f, indent=2)
    results["report_path"] = report_path
    print(f"[Evaluation] Report saved -> {report_path}")

    return results


def evaluate_classification_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
) -> Dict:
    """
    Evaluate nearest-prototype classification accuracy.

    Each test embedding is classified as the identity whose prototype
    has the highest cosine similarity to it.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        L2-normalised test embeddings.
    labels : np.ndarray, shape (N,)
        True integer labels.
    prototypes : np.ndarray, shape (C, D)
        L2-normalised prototype matrix (one row per class).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1_score, per_class_accuracy
    """
    similarities = embeddings @ prototypes.T   # (N, C)
    predicted = np.argmax(similarities, axis=1)

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    overall_acc = float(accuracy_score(labels, predicted))
    macro_prec = float(precision_score(labels, predicted, average="macro", zero_division=0))
    macro_rec = float(recall_score(labels, predicted, average="macro", zero_division=0))
    macro_f1 = float(f1_score(labels, predicted, average="macro", zero_division=0))

    unique_classes = np.unique(labels)
    per_class = {}
    for c in unique_classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_acc = float((predicted[mask] == labels[mask]).mean())
        per_class[int(c)] = class_acc

    results = {
        "accuracy": overall_acc,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1_score": macro_f1,
        "per_class_accuracy": per_class,
    }

    print(f"\n[Classification] Prototype-based Accuracy: {overall_acc:.4f} | F1: {macro_f1:.4f}")
    for c, acc in per_class.items():
        print(f"  Class {c}: {acc:.4f}")

    return results


def _plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    eer: float,
    eer_idx: int,
    save_path: str,
) -> None:
    """Internal helper to save a ROC curve plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random")
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", zorder=5,
               label=f"EER = {eer:.4f}")
    ax.set_xlabel("False Positive Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TAR)", fontsize=12)
    ax.set_title("Face Verification ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
