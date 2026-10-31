"""
train_demo.py
=============
Lightweight demonstration training script optimized for CPU execution.

Demonstrates all core research contributions in ~15-20 seconds without requiring
the multi-gigabyte CASIA-WebFace dataset:
1. Adaptive Empirical Prototype Memory initialization and dynamic updates.
2. Adaptive Margin Loss responding to sample difficulty.
3. Quality-aware sample confidence weighting.
4. Open-set verification evaluation (ROC, AUC, EER).
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from src.face_model import create_embedding_model
from src.models.prototype_memory import PrototypeMemory
from src.training.losses import AdaptiveMarginLoss
from src.training.trainer import QualityAwareTrainer
from src.evaluation.evaluate import evaluate_verification, evaluate_classification_accuracy


def run_cpu_demo_training(num_classes: int = 3, samples_per_class: int = 8, epochs: int = 2):
    print("=" * 65)
    print("  QUALITY-AWARE FACE RECOGNITION — CPU DEMONSTRATION TRAINING")
    print("=" * 65)
    print("Environment: Windows (CPU Execution)")
    print(f"Dataset Mode: Synthetic In-Memory Multi-Identity ({num_classes} IDs, {samples_per_class} img/ID)")
    print("Research Objectives Demonstrated:")
    print("  [Obj 1 & 2] Adaptive Empirical Prototypes & EMA Dynamic Updates")
    print("  [Obj 3]     Adaptive Angular Margins based on sample similarity")
    print("  [Obj 4]     Sample Quality and Confidence Loss Weighting")
    print("  [Obj 5]     Open-Set Verification (ROC, AUC, EER, TAR@FAR)")
    print("=" * 65)

    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Synthesize face images and ground-truth identities
    print("\n[Step 1/5] Synthesizing multi-identity face batches...")
    total_samples = num_classes * samples_per_class
    images = np.random.uniform(0.0, 1.0, size=(total_samples, 224, 224, 3)).astype(np.float32)
    labels = np.repeat(np.arange(num_classes, dtype=np.int32), samples_per_class)

    # Simulate varied sample quality: sharp well-lit vs blurry/noisy
    quality_scores = np.random.uniform(0.3, 0.95, size=(total_samples,)).astype(np.float32)

    # One-hot encode for training
    one_hot_labels = tf.one_hot(labels, depth=num_classes)

    # Build tf.data.Dataset
    batch_size = 6
    ds = tf.data.Dataset.from_tensor_slices((images, one_hot_labels))
    ds = ds.shuffle(buffer_size=total_samples, seed=42).batch(batch_size)

    # 2. Build MobileNetV2 Embedding Model (512-d output)
    print("\n[Step 2/5] Initializing MobileNetV2 512-dimensional Embedding Head...")
    # Using lightweight backbone config for fast CPU demo
    embedding_model = create_embedding_model(
        embedding_dim=512,
        backbone_trainable=False,
    )
    print("  Embedding architecture: MobileNetV2 -> GAP -> Dense(512) -> L2 Normalization")

    # 3. Setup Prototype Memory & Adaptive Loss
    print("\n[Step 3/5] Instantiating Adaptive Prototype Memory & Margin Loss...")
    proto_memory = PrototypeMemory(
        num_classes=num_classes,
        embedding_dim=512,
        momentum=0.15,
    )

    loss_fn = AdaptiveMarginLoss(
        num_classes=num_classes,
        embedding_dim=512,
        scale=30.0,
        base_margin=0.35,
        adaptive_lambda=0.5,
        margin_min=0.1,
        margin_max=0.7,
        noise_weight_floor=0.1,
    )

    # 4. Initialize Trainer & Run Training Epochs
    print("\n[Step 4/5] Running Quality-Aware Training on CPU...")
    trainer = QualityAwareTrainer(
        embedding_model=embedding_model,
        loss_fn=loss_fn,
        prototype_memory=proto_memory,
        num_classes=num_classes,
        learning_rate=1e-4,
        results_dir="results",
    )

    # Initial prototype gathering
    first_batch_imgs, first_batch_lbls = next(iter(ds))
    init_embs = embedding_model(first_batch_imgs, training=False).numpy()
    init_int_lbls = np.argmax(first_batch_lbls.numpy(), axis=1)
    proto_memory.initialise_from_embeddings(init_embs, init_int_lbls)
    print(f"  [Prototypes] Initialized class prototypes from batch: count={proto_memory.total_updates()}")

    start_t = time.time()
    for ep in range(1, epochs + 1):
        epoch_losses, epoch_accs = [], []
        step = 0
        for b_imgs, b_lbls in ds:
            loss_val, acc_val = trainer.train_single_step(b_imgs, b_lbls)
            epoch_losses.append(loss_val)
            epoch_accs.append(acc_val)
            step += 1

        mean_loss = float(np.mean(epoch_losses))
        mean_acc = float(np.mean(epoch_accs))
        print(f"  Epoch {ep:02d}/{epochs:02d} | Loss: {mean_loss:.4f} | Accuracy: {mean_acc:.4f} | Updates: {proto_memory.total_updates()}")

    elapsed = time.time() - start_t
    print(f"  Training finished in {elapsed:.2f} seconds on CPU.")

    # Save demo checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    ckpt_weights = os.path.join("checkpoints", "demo_embedding.weights.h5")
    proto_save_path = os.path.join("models", "demo_prototypes.npy")

    embedding_model.save_weights(ckpt_weights)
    proto_memory.save(proto_save_path)
    print(f"  Weights saved -> {ckpt_weights}")
    print(f"  Prototypes saved -> {proto_save_path}")

    # 5. Run Verification Evaluation on Extracted Embeddings
    print("\n[Step 5/5] Evaluating Verification Metrics (ROC, AUC, EER)...")
    eval_embs = embedding_model.predict(images, verbose=0)
    verif_results = evaluate_verification(
        embeddings=eval_embs,
        labels=labels,
        max_pairs_per_class=20,
        output_dir="reports",
        save_plots=True,
    )

    print("\n" + "=" * 65)
    print("  CPU DEMONSTRATION TRAINING COMPLETED SUCCESSFULLY")
    print(f"  AUC: {verif_results['auc']:.4f} | EER: {verif_results['eer']:.4f}")
    print("=" * 65)
    return verif_results


if __name__ == "__main__":
    run_cpu_demo_training()
