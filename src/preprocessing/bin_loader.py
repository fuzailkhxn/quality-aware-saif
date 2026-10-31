"""
bin_loader.py
=============
Loads InsightFace-format binary verification benchmark files
(.bin) for face verification evaluation.

Supported benchmarks (all included in faces_webface_112x112):
  - lfw.bin        (Labeled Faces in the Wild)
  - agedb_30.bin   (AgeDB-30, age variation)
  - calfw.bin      (Cross-Age LFW)
  - cfp_ff.bin     (CFP Frontal-Frontal)
  - cfp_fp.bin     (CFP Frontal-Profile, pose variation)
  - cplfw.bin      (Cross-Pose LFW)

Binary format (plain English)
------------------------------
Each .bin file is a Python pickle containing a tuple:
  (data_list, issame_list)

  data_list   : list of numpy arrays, each shape (3, 112, 112) or (112, 112, 3)
                These are pre-aligned 112×112 face images.
                Images come in PAIRS: data_list[2i] and data_list[2i+1]
                are the two faces for the i-th verification pair.

  issame_list : list of booleans (or 0/1 integers)
                issame_list[i] == True  → pair i is the SAME person
                issame_list[i] == False → pair i is DIFFERENT persons

So the evaluation protocol:
  For each pair (img_A, img_B):
    1. Embed both images using our model → emb_A, emb_B
    2. Compute cosine similarity score = dot(emb_A, emb_B)
    3. Compare score vs threshold to decide same/different
  Report AUC, EER, TAR@FAR across all pairs.
"""

import os
import pickle
import numpy as np
from typing import Tuple, List, Optional
import tensorflow as tf


# ===================================================================
# Load raw .bin data
# ===================================================================

def load_bin(
    bin_path: str,
    image_size: Tuple[int, int] = (112, 112),
) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Load an InsightFace .bin verification benchmark file.

    Parameters
    ----------
    bin_path   : path to the .bin file (e.g. 'lfw.bin')
    image_size : expected (height, width) — should be (112, 112) for CASIA

    Returns
    -------
    images   : list of np.ndarray, each shape (112, 112, 3), dtype float32,
               values in [0, 1].  Length = 2 × number_of_pairs.
    issame   : list of bool.  Length = number_of_pairs.
               issame[i] == True → images[2i] and images[2i+1] are same person.

    Raises
    ------
    FileNotFoundError : if bin_path does not exist
    ValueError        : if the file format is unexpected
    """
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Benchmark file not found: {bin_path}")

    print(f"[BinLoader] Loading: {bin_path}")
    with open(bin_path, "rb") as fh:
        data_list, issame_list = pickle.load(fh, encoding="bytes")

    images: List[np.ndarray] = []
    for raw in data_list:
        arr = np.array(raw, dtype=np.float32)

        # Handle both (C, H, W) and (H, W, C) formats
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)  # (3, H, W) → (H, W, 3)

        # Normalise pixel range to [0, 1]
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Resize if needed (rarely necessary; stored as 112×112)
        if arr.shape[:2] != image_size:
            from PIL import Image as _PIL
            pil_img = _PIL.fromarray((arr * 255).astype(np.uint8))
            pil_img = pil_img.resize((image_size[1], image_size[0]))
            arr = np.array(pil_img, dtype=np.float32) / 255.0

        images.append(arr)

    issame = [bool(s) for s in issame_list]

    n_pairs = len(issame)
    print(f"[BinLoader] Loaded {n_pairs} pairs ({len(images)} images) "
          f"from {os.path.basename(bin_path)}")
    return images, issame


# ===================================================================
# Embed all images in a bin file
# ===================================================================

def embed_bin_images(
    embedding_model: tf.keras.Model,
    images: List[np.ndarray],
    model_input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run all images through the embedding model in mini-batches.

    Parameters
    ----------
    embedding_model  : trained Keras model, output shape (None, embedding_dim)
    images           : list of np.ndarray shape (H, W, 3), values [0, 1]
    model_input_size : (H, W) that the model expects (may differ from 112×112)
    batch_size       : images per forward pass

    Returns
    -------
    embeddings : np.ndarray shape (len(images), embedding_dim)
                 L2-normalised (the embedding head already normalises, but
                 we normalise again for safety)
    """
    all_embs = []
    n = len(images)

    for start in range(0, n, batch_size):
        batch_imgs = images[start: start + batch_size]

        # Resize to model input size if needed
        if batch_imgs[0].shape[:2] != model_input_size:
            from PIL import Image as _PIL
            resized = []
            for img in batch_imgs:
                pil_img = _PIL.fromarray((img * 255).astype(np.uint8))
                pil_img = pil_img.resize((model_input_size[1], model_input_size[0]))
                resized.append(np.array(pil_img, dtype=np.float32) / 255.0)
            batch_imgs = resized

        batch_tensor = tf.constant(np.stack(batch_imgs, axis=0), dtype=tf.float32)
        embs = embedding_model(batch_tensor, training=False).numpy()

        # L2-normalise
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms

        all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)


# ===================================================================
# Evaluate one benchmark
# ===================================================================

def evaluate_bin_benchmark(
    embedding_model: tf.keras.Model,
    bin_path: str,
    output_dir: str,
    benchmark_label: str,
    model_input_size: Tuple[int, int] = (224, 224),
    far_thresholds: Optional[List[float]] = None,
    batch_size: int = 64,
) -> dict:
    """
    Full pipeline: load .bin → embed → compute verification metrics.

    Calls the existing evaluate_verification() function from
    src/evaluation/evaluate.py to report ROC, AUC, EER, TAR@FAR.

    Parameters
    ----------
    embedding_model  : trained Keras model
    bin_path         : path to .bin benchmark file
    output_dir       : where to save ROC curve plots and reports
    benchmark_label  : short name for the benchmark (e.g. 'lfw', 'agedb_30')
    model_input_size : (H, W) expected by the embedding model
    far_thresholds   : list of FAR values for TAR@FAR (e.g. [0.001, 0.01, 0.1])
    batch_size       : images per forward pass

    Returns
    -------
    metrics : dict with keys: auc, eer, tar_at_far, threshold
    """
    from src.evaluation.evaluate import evaluate_verification

    if far_thresholds is None:
        far_thresholds = [0.001, 0.01, 0.1]

    # Load images and ground-truth pairs
    images, issame = load_bin(bin_path, image_size=(112, 112))

    # Get embeddings for every image
    print(f"[BinLoader] Embedding {len(images)} images for '{benchmark_label}'...")
    embeddings = embed_bin_images(
        embedding_model, images,
        model_input_size=model_input_size,
        batch_size=batch_size,
    )

    # Build (emb_A, emb_B, label) pairs
    n_pairs = len(issame)
    emb_a = embeddings[0::2][:n_pairs]   # even indices
    emb_b = embeddings[1::2][:n_pairs]   # odd indices
    pair_labels = np.array([1 if s else 0 for s in issame], dtype=np.int32)

    # Compute cosine similarity scores
    # dot product of L2-normed vectors = cosine similarity
    scores = np.sum(emb_a * emb_b, axis=1)  # shape (n_pairs,)

    # Use evaluate_verification from evaluate.py
    # It expects embeddings + integer labels (one label per IMAGE, not per pair).
    # We pass it the pre-computed pair scores directly via a wrapper.
    os.makedirs(output_dir, exist_ok=True)
    metrics = _evaluate_from_pair_scores(
        scores=scores,
        issame=issame,
        benchmark_label=benchmark_label,
        output_dir=output_dir,
        far_thresholds=far_thresholds,
    )

    return metrics


def _evaluate_from_pair_scores(
    scores: np.ndarray,
    issame: List[bool],
    benchmark_label: str,
    output_dir: str,
    far_thresholds: List[float],
) -> dict:
    """
    Compute and print verification metrics from pre-computed cosine similarity
    scores and ground-truth same/different labels.

    This is a standalone implementation that does NOT require the
    evaluate.py generate_pairs step (we already have explicit pairs from .bin).

    Parameters
    ----------
    scores    : cosine similarity for each pair, shape (n_pairs,)
    issame    : True = same person, False = different
    benchmark_label : name of the benchmark (for printing/saving)
    output_dir      : where to save ROC plot
    far_thresholds  : FAR values for TAR@FAR reporting

    Returns
    -------
    dict with auc, eer, tar_at_far dict
    """
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_bin = np.array([1 if s else 0 for s in issame], dtype=np.int32)

    # ROC curve  (higher score = more likely same person)
    fpr, tpr, thresholds = roc_curve(labels_bin, scores, pos_label=1)
    roc_auc = sklearn_auc(fpr, tpr)

    # EER: point where FAR ≈ FRR  (FRR = 1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))

    # TAR @ FAR thresholds
    tar_at_far = {}
    for target_far in far_thresholds:
        idx = np.searchsorted(fpr, target_far, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        tar_at_far[target_far] = float(tpr[idx])

    # Best threshold (maximise TPR - FPR i.e. Youden's J)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    # Print results
    print(f"\n{'='*55}")
    print(f"  Verification Results — {benchmark_label.upper()}")
    print(f"{'='*55}")
    print(f"  AUC  : {roc_auc:.4f}")
    print(f"  EER  : {eer:.4f}  ({eer*100:.2f}%)")
    for far, tar in tar_at_far.items():
        print(f"  TAR @ FAR={far:.3f} : {tar:.4f}  ({tar*100:.2f}%)")
    print(f"  Best threshold : {best_threshold:.4f}")
    print(f"{'='*55}\n")

    # Save ROC curve plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR)")
    ax.set_title(f"ROC — {benchmark_label.upper()}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, f"roc_{benchmark_label}.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[BinLoader] ROC plot saved → {plot_path}")

    # Save metrics text report
    report_path = os.path.join(output_dir, f"report_{benchmark_label}.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Benchmark : {benchmark_label}\n")
        fh.write(f"N pairs   : {len(issame)}\n")
        fh.write(f"AUC       : {roc_auc:.6f}\n")
        fh.write(f"EER       : {eer:.6f}\n")
        for far, tar in tar_at_far.items():
            fh.write(f"TAR@FAR={far:.3f} : {tar:.6f}\n")
        fh.write(f"Best threshold : {best_threshold:.6f}\n")
    print(f"[BinLoader] Report saved → {report_path}")

    return {
        "benchmark": benchmark_label,
        "auc": roc_auc,
        "eer": eer,
        "tar_at_far": tar_at_far,
        "best_threshold": best_threshold,
    }


# ===================================================================
# Evaluate ALL available benchmarks
# ===================================================================

def evaluate_all_benchmarks(
    embedding_model: tf.keras.Model,
    benchmark_dir: str,
    output_dir: str,
    model_input_size: Tuple[int, int] = (224, 224),
    far_thresholds: Optional[List[float]] = None,
    batch_size: int = 64,
) -> List[dict]:
    """
    Evaluate the model on every .bin benchmark found in benchmark_dir.

    Parameters
    ----------
    embedding_model : trained Keras model
    benchmark_dir   : directory containing lfw.bin, agedb_30.bin, etc.
    output_dir      : where to save ROC plots and reports
    model_input_size: (H, W) the model expects
    far_thresholds  : list of FAR values
    batch_size      : forward pass batch size

    Returns
    -------
    results : list of metric dicts (one per benchmark)
    """
    if far_thresholds is None:
        far_thresholds = [0.001, 0.01, 0.1]

    known_benchmarks = [
        "lfw", "agedb_30", "calfw", "cfp_ff", "cfp_fp", "cplfw"
    ]

    results = []
    for name in known_benchmarks:
        bin_path = os.path.join(benchmark_dir, f"{name}.bin")
        if not os.path.isfile(bin_path):
            print(f"[BinLoader] Skipping '{name}' — file not found: {bin_path}")
            continue
        try:
            metrics = evaluate_bin_benchmark(
                embedding_model=embedding_model,
                bin_path=bin_path,
                output_dir=output_dir,
                benchmark_label=name,
                model_input_size=model_input_size,
                far_thresholds=far_thresholds,
                batch_size=batch_size,
            )
            results.append(metrics)
        except Exception as exc:
            print(f"[BinLoader] ERROR evaluating {name}: {exc}")

    return results

