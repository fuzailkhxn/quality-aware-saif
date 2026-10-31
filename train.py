"""
train.py
=========
Main training entry point for the Quality-Aware Face Recognition system.

Usage
-----
# Train with quality-aware system (default):
    python train.py

# Train with original baseline model (for comparison):
    python train.py --baseline

# Override dataset path:
    python train.py --dataset path/to/dataset

# Override number of epochs:
    python train.py --epochs 20

How the training works (plain English)
----------------------------------------
1. Images are loaded from dataset/  (one subfolder per person).
2. Quality scores are computed for each image (sharpness + brightness).
3. The embedding model maps each face image to a 512-number vector.
4. The adaptive margin loss trains the model so same-person vectors cluster
   together and different-person vectors push apart.
5. Prototypes (average face vectors per person) are updated after each batch.
6. After training, evaluation metrics (accuracy, ROC, AUC, EER) are reported.
7. Model weights and prototypes are saved for later inference.

Dataset format expected
-----------------------
dataset/
  person_A/
    img1.jpg
    img2.jpg
    ...
  person_B/
    img1.jpg
    ...
  (minimum 2 people, minimum ~10 images per person recommended)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import Optional
# -------------------------------------------------------------------------
# Make sure project root is in Python path (for src.* imports to work)
# -------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from src.utils.seed_utils import set_global_seed
from src.face_model import create_face_model, create_embedding_model
from src.models.prototype_memory import PrototypeMemory
from src.training.losses import build_adaptive_margin_loss
from src.training.trainer import QualityAwareTrainer
from src.evaluation.evaluate import evaluate_verification, evaluate_classification_accuracy
from src.preprocessing.dataset_loader import (
    scan_and_validate_dataset,
    create_training_and_verification_splits,
    build_tf_dataset_from_dict,
    is_recordio_dataset,
)


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Quality-Aware Face Recognition model"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train the original baseline classifier instead of quality-aware model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (overrides config.DATASET_PATH)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config.EPOCHS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help=f"Random seed (default: {cfg.SEED})",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=cfg.MAX_IDENTITIES,
        help=f"Limit number of identities (default: {cfg.MAX_IDENTITIES})",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=cfg.MIN_IMAGES_PER_IDENTITY,
        help=f"Minimum images per identity (default: {cfg.MIN_IMAGES_PER_IDENTITY})",
    )
    parser.add_argument(
        "--recordio",
        action="store_true",
        help=(
            "Use MXNet RecordIO dataset (train.rec + train.idx + train.lst) "
            "instead of the default folder-based dataset. "
            "Set the dataset directory with --dataset or config.RECORDIO_DATASET_PATH."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help=(
            "Directory containing .bin verification benchmarks "
            "(lfw.bin, agedb_30.bin, etc.). "
            "Defaults to the same directory as --dataset in RecordIO mode."
        ),
    )
    parser.add_argument(
        "--test-loader",
        action="store_true",
        help="Test dataset loading without training (draws 1 batch and verifies shapes and classes)",
    )
    parser.add_argument(
        "--test-step",
        action="store_true",
        help="Test exactly one training step (forward + loss + backprop + optimizer + prototype update)",
    )
    return parser.parse_args()


# -------------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------------

def build_dataset(
    dataset_path: str,
    validation_split: float,
    image_size: tuple,
    batch_size: int,
    augment: bool = True,
    max_identities: Optional[int] = None,
    min_images_per_identity: int = 10,
    seed: int = 42,
):
    """
    Build high-performance tf.data.Dataset objects for training and validation.
    Automatically detects whether dataset_path is an MXNet RecordIO or folder-based dataset.
    """
    if is_recordio_dataset(dataset_path):
        print(f"[Dataset] Auto-detected MXNet RecordIO dataset at: {dataset_path}")
        from src.preprocessing.recordio_loader import load_recordio_splits
        rec_path = os.path.join(dataset_path, "train.rec")
        idx_path = os.path.join(dataset_path, "train.idx")
        lst_path = os.path.join(dataset_path, "train.lst")
        train_ds, val_ds, test_ds, num_classes, train_classes = load_recordio_splits(
            rec_path=rec_path,
            idx_path=idx_path,
            lst_path=lst_path,
            image_size=image_size,
            batch_size=batch_size,
            max_identities=max_identities or cfg.MAX_IDENTITIES,
            min_images=min_images_per_identity,
            val_fraction=validation_split,
            test_fraction=cfg.RECORDIO_TEST_FRACTION,
            seed=seed,
        )
        return train_ds, val_ds, test_ds, train_classes, num_classes

    dataset_dict = scan_and_validate_dataset(
        dataset_path=dataset_path,
        min_images_per_identity=min_images_per_identity,
        max_identities=max_identities,
        seed=seed,
    )

    train_dict, val_dict, test_verification_dict = create_training_and_verification_splits(
        dataset_dict=dataset_dict,
        test_identity_fraction=cfg.VAL_IDENTITY_FRACTION,
        train_val_image_fraction=validation_split,
        seed=seed,
    )

    train_ds, train_classes, num_classes = build_tf_dataset_from_dict(
        train_dict,
        image_size=image_size,
        batch_size=batch_size,
        augment=augment,
        is_training=True,
        seed=seed,
    )

    val_ds, _, _ = build_tf_dataset_from_dict(
        val_dict,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        is_training=False,
        seed=seed,
    )

    test_ds, _, _ = build_tf_dataset_from_dict(
        test_verification_dict,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        is_training=False,
        seed=seed,
    )

    return train_ds, val_ds, test_ds, train_classes, num_classes


# -------------------------------------------------------------------------
# Baseline training (original simple classifier)
# -------------------------------------------------------------------------

def train_baseline(
    dataset_path: str,
    epochs: int,
    seed: int,
) -> None:
    """
    Train the original baseline MobileNetV2 classifier.

    This is kept for comparison purposes. Uses the original create_face_model()
    function and Keras model.fit() without quality-aware components.
    """
    print("\n" + "=" * 60)
    print("  Baseline Training (Original Model)")
    print("=" * 60)

    if is_recordio_dataset(dataset_path):
        train_data, val_data, _, _, num_classes = build_dataset(
            dataset_path=dataset_path,
            validation_split=cfg.VALIDATION_SPLIT,
            image_size=cfg.IMAGE_SIZE,
            batch_size=cfg.BATCH_SIZE,
            augment=True,
            max_identities=cfg.MAX_IDENTITIES,
            min_images_per_identity=cfg.MIN_IMAGES_PER_IDENTITY,
            seed=seed,
        )
        print(f"[Baseline] Using RecordIO dataset with {num_classes} classes")
    else:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=15,
            horizontal_flip=True,
        )

        train_data = datagen.flow_from_directory(
            dataset_path,
            target_size=cfg.IMAGE_SIZE,
            batch_size=cfg.BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            seed=seed,
        )

        val_data = datagen.flow_from_directory(
            dataset_path,
            target_size=cfg.IMAGE_SIZE,
            batch_size=cfg.BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            seed=seed,
        )

        num_classes = len(train_data.class_indices)
        print(f"[Baseline] Classes: {train_data.class_indices}")

    model = create_face_model(num_classes=num_classes)

    os.makedirs("models", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            cfg.MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(cfg.RESULTS_DIR, "logs", "baseline_log.csv"),
        ),
    ]

    os.makedirs(os.path.join(cfg.RESULTS_DIR, "logs"), exist_ok=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(cfg.MODEL_SAVE_PATH)
    print(f"\n[Baseline] Model saved → {cfg.MODEL_SAVE_PATH}")
    return history


# -------------------------------------------------------------------------
# Quality-aware training
# -------------------------------------------------------------------------

def train_quality_aware(
    dataset_path: str,
    epochs: int,
    seed: int,
    max_identities: Optional[int] = None,
    min_images: int = 10,
    run_eval: bool = True,
) -> dict:
    """
    Train using the full quality-aware pipeline:
      1. Embedding model (MobileNetV2 + 512-d L2 normalised head)
      2. Adaptive margin loss (dynamic margins + noisy sample weighting)
      3. Prototype memory (updated after each batch)
      4. Evaluation (ROC, AUC, EER, TAR@FAR on open-set held-out identities)
    """
    print("\n" + "=" * 60)
    print("  Quality-Aware Training")
    print("=" * 60)

    # Build datasets with corrupt checks and disjoint splits
    train_ds, val_ds, test_ds, class_names, num_classes = build_dataset(
        dataset_path=dataset_path,
        validation_split=cfg.VALIDATION_SPLIT,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        augment=True,
        max_identities=max_identities,
        min_images_per_identity=min_images,
        seed=seed,
    )

    print(f"\n[Model] Building embedding model (dim={cfg.EMBEDDING_DIM})...")
    embedding_model = create_embedding_model(embedding_dim=cfg.EMBEDDING_DIM)

    print(f"[Loss]  Building adaptive margin loss ({num_classes} classes)...")
    loss_fn = build_adaptive_margin_loss(num_classes, cfg)

    print(f"[Proto] Building prototype memory...")
    prototype_memory = PrototypeMemory(
        num_classes=num_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        momentum=1.0 - cfg.PROTOTYPE_MOMENTUM,  # convert: momentum=0.9 → α=0.1
    )

    trainer = QualityAwareTrainer(
        embedding_model=embedding_model,
        loss_fn=loss_fn,
        prototype_memory=prototype_memory,
        num_classes=num_classes,
        learning_rate=cfg.LEARNING_RATE,
        results_dir=cfg.RESULTS_DIR,
        prototype_init_batches=cfg.PROTOTYPE_INIT_BATCHES,
        quality_blur_threshold=cfg.BLUR_THRESHOLD,
    )

    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        patience=7,
        model_save_path=cfg.EMBEDDING_WEIGHTS_PATH,
        prototype_save_path=cfg.PROTOTYPE_SAVE_PATH,
    )

    # Save full model (not just weights)
    os.makedirs("models", exist_ok=True)
    embedding_model.save(cfg.EMBEDDING_MODEL_PATH)
    print(f"[Save] Full embedding model saved -> {cfg.EMBEDDING_MODEL_PATH}")

    # ----------------------------------------------------------------
    # Evaluation on UNSEEN identities (Zero Data Leakage)
    # ----------------------------------------------------------------
    if run_eval:
        print("\n[Eval] Running open-set face verification evaluation on held-out identities...")
        _run_evaluation(
            embedding_model=embedding_model,
            prototype_memory=prototype_memory,
            val_dataset=test_ds,
        )

    return history


def _run_evaluation(
    embedding_model: tf.keras.Model,
    prototype_memory: PrototypeMemory,
    val_dataset: tf.data.Dataset,
) -> None:
    """Collect validation embeddings and run full evaluation."""
    all_embeddings, all_labels = [], []

    for images, labels in val_dataset:
        embs = embedding_model(images, training=False).numpy()
        int_labels = np.argmax(labels.numpy(), axis=1)
        all_embeddings.append(embs)
        all_labels.append(int_labels)

    if not all_embeddings:
        print("[Eval] No validation data available for evaluation.")
        return

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print("[Eval] Need at least 2 classes in validation set for verification eval.")
        return

    n_per_class = {int(c): int((labels == c).sum()) for c in unique_classes}
    has_multi = any(v >= 2 for v in n_per_class.values())
    if not has_multi:
        print("[Eval] Each class has only 1 validation image — cannot compute genuine pairs.")
        print("[Eval] Skipping verification evaluation. Add more images per person.")
    else:
        os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
        evaluate_verification(
            embeddings=embeddings,
            labels=labels,
            far_thresholds=cfg.FAR_THRESHOLDS,
            output_dir=cfg.REPORTS_DIR,
        )

    if prototype_memory.is_initialised():
        prototypes = prototype_memory.get_prototypes()
        evaluate_classification_accuracy(embeddings, labels, prototypes)


# -------------------------------------------------------------------------
# RecordIO quality-aware training
# -------------------------------------------------------------------------

def train_quality_aware_recordio(
    dataset_dir: str,
    benchmark_dir: str,
    epochs: int,
    seed: int,
    max_identities: int = 100,
    min_images: int = 10,
    run_eval: bool = True,
) -> dict:
    """
    Train the full quality-aware pipeline using CASIA-WebFace RecordIO format.

    Instead of reading images from a folder tree, this function reads
    train.rec / train.idx / train.lst directly.  All five research objectives
    (adaptive prototypes, dynamic updates, adaptive margins, noisy sample
    handling, and face verification) are preserved.

    Parameters
    ----------
    dataset_dir    : directory containing train.rec, train.idx, train.lst
    benchmark_dir  : directory containing lfw.bin, agedb_30.bin, etc.
    epochs         : number of training epochs
    seed           : random seed for reproducibility
    max_identities : maximum number of identities to use (for fast experiments)
    min_images     : skip identities with fewer images than this
    run_eval       : whether to run verification evaluation after training

    Plain-English flow
    ------------------
    1. Parse train.lst → identity→record mapping.
    2. Parse train.idx → record→byte-offset mapping.
    3. Select top `max_identities` identities.
    4. Split into identity-disjoint train / val / test sets.
    5. Build tf.data.Dataset pipelines that lazily read from train.rec.
    6. Train with the same QualityAwareTrainer used in folder mode.
    7. Evaluate using .bin benchmark files (lfw, agedb_30, calfw, cfp, cplfw).
    """
    from src.preprocessing.recordio_loader import load_recordio_splits
    from src.preprocessing.bin_loader import evaluate_all_benchmarks

    print("\n" + "=" * 60)
    print("  Quality-Aware Training (RecordIO mode)")
    print("=" * 60)

    # Paths
    rec_path = os.path.join(dataset_dir, "train.rec")
    idx_path = os.path.join(dataset_dir, "train.idx")
    lst_path = os.path.join(dataset_dir, "train.lst")

    for p, name in [(rec_path, "train.rec"), (idx_path, "train.idx"), (lst_path, "train.lst")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Please check that RECORDIO_DATASET_PATH in config.py is correct."
            )

    # Build identity-disjoint tf.data.Dataset splits
    print(f"\n[RecordIO] Loading dataset from: {dataset_dir}")
    train_ds, val_ds, test_ds, num_classes, train_labels = load_recordio_splits(
        rec_path=rec_path,
        idx_path=idx_path,
        lst_path=lst_path,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        max_identities=max_identities,
        min_images=min_images,
        val_fraction=cfg.VAL_IDENTITY_FRACTION,
        test_fraction=cfg.RECORDIO_TEST_FRACTION,
        seed=seed,
    )

    print(f"\n[Model] Building embedding model (dim={cfg.EMBEDDING_DIM})...")
    embedding_model = create_embedding_model(embedding_dim=cfg.EMBEDDING_DIM)

    print(f"[Loss]  Building adaptive margin loss ({num_classes} classes)...")
    loss_fn = build_adaptive_margin_loss(num_classes, cfg)

    print(f"[Proto] Building prototype memory...")
    prototype_memory = PrototypeMemory(
        num_classes=num_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        momentum=1.0 - cfg.PROTOTYPE_MOMENTUM,
    )

    trainer = QualityAwareTrainer(
        embedding_model=embedding_model,
        loss_fn=loss_fn,
        prototype_memory=prototype_memory,
        num_classes=num_classes,
        learning_rate=cfg.LEARNING_RATE,
        results_dir=cfg.RESULTS_DIR,
        prototype_init_batches=cfg.PROTOTYPE_INIT_BATCHES,
        quality_blur_threshold=cfg.BLUR_THRESHOLD,
    )

    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        patience=7,
        model_save_path=cfg.EMBEDDING_WEIGHTS_PATH,
        prototype_save_path=cfg.PROTOTYPE_SAVE_PATH,
    )

    # Save full model
    os.makedirs("models", exist_ok=True)
    embedding_model.save(cfg.EMBEDDING_MODEL_PATH)
    print(f"[Save] Embedding model saved -> {cfg.EMBEDDING_MODEL_PATH}")

    # ------------------------------------------------------------------
    # Evaluation using .bin benchmark files
    # ------------------------------------------------------------------
    if run_eval:
        print("\n[Eval] Running face verification on .bin benchmarks...")
        os.makedirs(cfg.REPORTS_DIR, exist_ok=True)

        results = evaluate_all_benchmarks(
            embedding_model=embedding_model,
            benchmark_dir=benchmark_dir,
            output_dir=cfg.REPORTS_DIR,
            model_input_size=cfg.IMAGE_SIZE,
            far_thresholds=cfg.FAR_THRESHOLDS,
        )

        if results:
            print("\n[Eval] Summary:")
            print(f"  {'Benchmark':<12}  {'AUC':>7}  {'EER':>7}")
            print(f"  {'-'*30}")
            for r in results:
                print(f"  {r['benchmark']:<12}  {r['auc']:.4f}   {r['eer']:.4f}")

    return history


# -------------------------------------------------------------------------
# Loader verification test (no training)
# -------------------------------------------------------------------------

def test_dataset_loader(
    dataset_path: str,
    max_identities: Optional[int] = None,
    min_images: int = 10,
    seed: int = 42,
) -> None:
    """
    Test dataset loading: verifies batches can be drawn, shapes match expectations,
    and labels are properly formed. Does not perform model training.
    """
    print("\n" + "=" * 60)
    print("  Dataset Loader Verification Test")
    print("=" * 60)
    print(f"[Test] Path: {dataset_path}")
    is_rec = is_recordio_dataset(dataset_path)
    print(f"[Test] Format detected: {'MXNet RecordIO (.rec/.idx/.lst)' if is_rec else 'Folder-based'}")

    train_ds, val_ds, test_ds, train_classes, num_classes = build_dataset(
        dataset_path=dataset_path,
        validation_split=cfg.VALIDATION_SPLIT,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        augment=False,
        max_identities=max_identities,
        min_images_per_identity=min_images,
        seed=seed,
    )

    print(f"\n[Test] Number of training classes: {num_classes}")

    # Fetch 1 batch from train_ds
    print("\n[Test] Testing train_ds generator...")
    for images, labels in train_ds.take(1):
        print(f"  -> Images shape: {images.shape}, dtype: {images.dtype}, range: [{float(tf.reduce_min(images)):.2f}, {float(tf.reduce_max(images)):.2f}]")
        print(f"  -> Labels shape: {labels.shape}, dtype: {labels.dtype}")
        assert images.shape[1:3] == cfg.IMAGE_SIZE, f"Expected image size {cfg.IMAGE_SIZE}, got {images.shape[1:3]}"
        assert labels.shape[1] == num_classes, f"Expected {num_classes} classes, got {labels.shape[1]}"
        print("  -> train_ds batch verified successfully.")

    # Fetch 1 batch from val_ds
    print("\n[Test] Testing val_ds generator...")
    for images, labels in val_ds.take(1):
        print(f"  -> Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"  -> Labels shape: {labels.shape}, dtype: {labels.dtype}")
        assert labels.shape[1] == num_classes, f"Validation labels shape {labels.shape[1]} must match training classes {num_classes}"
        print("  -> val_ds batch verified successfully (matching training class dimension).")

    # Fetch 1 batch from test_ds
    print("\n[Test] Testing test_ds generator...")
    for images, labels in test_ds.take(1):
        print(f"  -> Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"  -> Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print("  -> test_ds batch verified successfully.")

    print("\n" + "=" * 60)
    print("  [SUCCESS] Dataset loader test passed! Ready for training.")
    print("=" * 60 + "\n")


def test_single_training_step(
    dataset_path: str,
    max_identities: Optional[int] = None,
    min_images: int = 10,
    seed: int = 42,
) -> None:
    """
    Execute exactly one training step to verify that forward pass, loss calculation,
    gradient tape, backpropagation, optimizer update, and prototype memory update
    all run without any errors. Does not train the full model.
    """
    print("\n" + "=" * 60)
    print("  Single Training Step Verification Test")
    print("=" * 60)
    print(f"[Step Test] Dataset path: {dataset_path}")

    train_ds, val_ds, test_ds, train_classes, num_classes = build_dataset(
        dataset_path=dataset_path,
        validation_split=cfg.VALIDATION_SPLIT,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        augment=False,
        max_identities=max_identities,
        min_images_per_identity=min_images,
        seed=seed,
    )

    print(f"[Step Test] Classes: {num_classes}")
    print("[Step Test] Building embedding model...")
    embedding_model = create_embedding_model(embedding_dim=cfg.EMBEDDING_DIM)

    print("[Step Test] Building adaptive margin loss...")
    loss_fn = build_adaptive_margin_loss(num_classes, cfg)

    print("[Step Test] Initialising prototype memory...")
    proto_mem = PrototypeMemory(
        num_classes=num_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        momentum=1.0 - cfg.PROTOTYPE_MOMENTUM,
    )

    trainer = QualityAwareTrainer(
        embedding_model=embedding_model,
        loss_fn=loss_fn,
        prototype_memory=proto_mem,
        num_classes=num_classes,
        learning_rate=cfg.LEARNING_RATE,
        results_dir=cfg.RESULTS_DIR,
    )

    print("[Step Test] Fetching 1 batch from train_ds...")
    protos_before = proto_mem.get_prototypes().copy()
    updates_before = proto_mem.total_updates()

    for images, labels in train_ds.take(1):
        loss_val, acc = trainer.train_single_step(images, labels)
        protos_after = proto_mem.get_prototypes().copy()
        updates_after = proto_mem.total_updates()
        delta = float(np.linalg.norm(protos_after - protos_before))
        int_lbls = np.argmax(labels.numpy(), axis=1)
        active_classes = sorted(list(set(int(c) for c in int_lbls)))
        updated_classes = [c for c in active_classes if proto_mem.has_prototype(c)]

        print(f"  -> Forward + backward step executed successfully!")
        print(f"  -> Step Loss: {loss_val:.4f}")
        print(f"  -> Step Batch Accuracy: {acc * 100:.2f}%")
        print(f"  -> Prototype Memory updated: {updates_after > updates_before} ({updates_after} samples updated across classes: {updated_classes})")
        print(f"  -> Prototype vector change (L2 delta): {delta:.4f}")
        print(f"  -> Prototype Memory initialised: {proto_mem.is_initialised()}")

        assert updates_after > updates_before, "Prototype updates must increase after train step"
        assert delta > 0.0, "Prototype vectors must change values after train step"
        break

    print("\n" + "=" * 60)
    print("  [SUCCESS] One training step verified! Prototypes updated successfully.")
    print("=" * 60 + "\n")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Set seeds before anything else
    set_global_seed(args.seed)

    epochs = args.epochs or cfg.EPOCHS

    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        if cfg.USE_RECORDIO or is_recordio_dataset(cfg.RECORDIO_DATASET_PATH):
            dataset_path = cfg.RECORDIO_DATASET_PATH
        else:
            dataset_path = cfg.get_abs_path(cfg.DATASET_PATH)

    # Automatic RecordIO detection: True if --recordio passed, config requested it, or files exist
    use_recordio = bool(args.recordio or cfg.USE_RECORDIO or is_recordio_dataset(dataset_path))
    detection_source = "flag --recordio" if args.recordio else ("config" if cfg.USE_RECORDIO else "auto-detected")

    print(f"\n[Config] Dataset path : {dataset_path}")
    print(f"[Config] Epochs       : {epochs}")
    print(f"[Config] Seed         : {args.seed}")
    print(f"[Config] Max IDs      : {args.max_identities}")
    print(f"[Config] Min Images   : {args.min_images}")
    print(f"[Config] Mode         : {'Baseline' if args.baseline else 'Quality-Aware'}")
    print(f"[Config] Dataset type : {'RecordIO (.rec) [' + detection_source + ']' if use_recordio else 'Folder'}")

    # Loader verification mode (exits without training)
    if args.test_loader:
        test_dataset_loader(
            dataset_path=dataset_path,
            max_identities=args.max_identities,
            min_images=args.min_images,
            seed=args.seed,
        )
        return

    # Single training step verification mode (exits after 1 step)
    if args.test_step:
        test_single_training_step(
            dataset_path=dataset_path,
            max_identities=args.max_identities,
            min_images=args.min_images,
            seed=args.seed,
        )
        return

    if args.baseline:
        train_baseline(dataset_path, epochs, args.seed)

    elif use_recordio:
        benchmark_dir = args.benchmark_dir or dataset_path
        print(f"[Config] Benchmark dir: {benchmark_dir}")
        train_quality_aware_recordio(
            dataset_dir=dataset_path,
            benchmark_dir=benchmark_dir,
            epochs=epochs,
            seed=args.seed,
            max_identities=args.max_identities,
            min_images=args.min_images,
            run_eval=not args.no_eval,
        )

    else:
        train_quality_aware(
            dataset_path=dataset_path,
            epochs=epochs,
            seed=args.seed,
            max_identities=args.max_identities,
            min_images=args.min_images,
            run_eval=not args.no_eval,
        )


if __name__ == "__main__":
    main()
