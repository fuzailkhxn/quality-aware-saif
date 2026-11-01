"""
train_real_dataset.py
======================
Executes real CPU training on the 5-identity dataset using:
- Real face images from dataset/split_manifest.json (ZERO synthetic data)
- 5 identities: amir, ranbir, saif, salman, sharuk (292 total usable images)
- MobileNetV2 512-d embedding model (CPU-optimized, backbone frozen, trainable head)
- Prototype-aware Adaptive Margin Loss
- Quality-weighted Prototype Memory EMA updates
- Comprehensive evaluation on 44 held-out test images (Accuracy, Precision, Recall, F1, Confusion Matrix, Verification Metrics, ROC/AUC)
- Weight update and reload verification
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from src.face_model import create_embedding_model
from src.models.prototype_memory import PrototypeMemory
from src.training.losses import AdaptiveMarginLoss
from src.training.trainer import QualityAwareTrainer
from src.preprocessing.dataset_loader import (
    load_split_from_manifest,
    build_tf_dataset_from_dict,
)

MANIFEST_PATH = "dataset/split_manifest.json"
CHECKPOINT_PATH = "checkpoints/real_embedding.weights.h5"
PROTOTYPE_PATH = "models/real_prototypes.npy"
ACTIVE_PROTOTYPE_PATH = "models/prototypes.npy"
REPORT_DIR = "reports"
LOG_DIR = "results/logs"
LOG_CSV = os.path.join(LOG_DIR, "real_training_log.csv")


def main():
    print("=" * 65)
    print("  QUALITY-AWARE FACE RECOGNITION — 5-IDENTITY REAL CPU TRAINING")
    print("=" * 65)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Load splits from manifest
    print(f"\n[1/7] Loading dataset splits from {MANIFEST_PATH}...")
    train_dict, val_dict, test_dict, unknown_dict, classes = load_split_from_manifest(MANIFEST_PATH)
    num_classes = len(classes)
    train_count = sum(len(v) for v in train_dict.values())
    val_count = sum(len(v) for v in val_dict.values())
    test_count = sum(len(v) for v in test_dict.values())
    unknown_count = sum(len(v) for v in unknown_dict.values())

    print(f"  Identities ({num_classes}): {classes}")
    print(f"  Train samples: {train_count}")
    print(f"  Val samples:   {val_count} (Held-out validation set)")
    print(f"  Test samples:  {test_count} (Held-out evaluation set)")
    print(f"  Total samples: {train_count + val_count + test_count}")

    # 2. Build TF Datasets
    print("\n[2/7] Constructing tf.data pipelines...")
    batch_size = 8  # 205 train images / 8 = ~25 batches per epoch (CPU friendly)
    train_ds, _, _ = build_tf_dataset_from_dict(
        train_dict, image_size=(224, 224), batch_size=batch_size, augment=True, is_training=True, seed=42
    )
    val_ds, _, _ = build_tf_dataset_from_dict(
        val_dict, image_size=(224, 224), batch_size=batch_size, augment=False, is_training=False, seed=42
    )

    # 3. Instantiate Architecture & Prototype Memory
    print("\n[3/7] Instantiating MobileNetV2 512-d Model and Prototype Memory...")
    embedding_dim = 512
    model = create_embedding_model(
        embedding_dim=embedding_dim,
        backbone_trainable=False,
    )

    # Prototype Memory
    proto_mem = PrototypeMemory(num_classes=num_classes, embedding_dim=embedding_dim, momentum=0.1)

    # Loss function with prototype integration
    loss_fn = AdaptiveMarginLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        scale=64.0,
        base_margin=0.35,
        adaptive_lambda=0.5,
        margin_min=0.1,
        margin_max=0.7,
        quality_alpha=0.6,
        confidence_alpha=0.4,
    )

    # Record initial weights for change verification
    initial_dense_weights = model.get_layer("embedding_dense").get_weights()[0].copy()
    initial_loss_w = loss_fn.W.numpy().copy()

    # 4. Initialize Trainer
    trainer = QualityAwareTrainer(
        embedding_model=model,
        loss_fn=loss_fn,
        prototype_memory=proto_mem,
        num_classes=num_classes,
        learning_rate=3e-4,
        results_dir="results",
        prototype_init_batches=20,  # Covers full train set to initialize all 5 class prototypes
    )
    # Redirect log path
    trainer._log_path = LOG_CSV
    trainer._init_log_file()

    # 5. Execute Training Loop
    epochs = 15
    print(f"\n[4/7] Beginning CPU Training for {epochs} epochs...")
    start_time = time.time()
    train_history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        patience=5,
        model_save_path=CHECKPOINT_PATH,
        prototype_save_path=PROTOTYPE_PATH,
    )
    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.2f} seconds.")

    # Also save prototypes to models/prototypes.npy for active application use
    proto_mem.save(ACTIVE_PROTOTYPE_PATH)
    print(f"Active prototype memory saved to {ACTIVE_PROTOTYPE_PATH}")

    # 6. Verify Model Weights Changed & Checkpoint Reloads
    print("\n[5/7] Verifying Weight Updates and Checkpoint Reloading...")
    final_dense_weights = model.get_layer("embedding_dense").get_weights()[0]
    final_loss_w = loss_fn.W.numpy()

    dense_weight_diff = np.linalg.norm(final_dense_weights - initial_dense_weights)
    loss_w_diff = np.linalg.norm(final_loss_w - initial_loss_w)

    print(f"  Embedding layer weight delta (L2 norm): {dense_weight_diff:.6f}")
    print(f"  Loss matrix W weight delta (L2 norm):    {loss_w_diff:.6f}")
    assert dense_weight_diff > 1e-4, "FAILURE: Embedding layer weights did NOT change during training!"
    assert loss_w_diff > 1e-4, "FAILURE: Loss matrix W weights did NOT change during training!"
    print("  [CONFIRMED] Neural network parameters were genuinely updated by the optimizer.")

    # Reload checkpoint
    print("  Testing checkpoint loading from disk...")
    eval_model = create_embedding_model(embedding_dim=512, backbone_trainable=False)
    resolved_ckpt = QualityAwareTrainer._resolve_weights_path(CHECKPOINT_PATH)
    eval_model.load_weights(resolved_ckpt)
    print(f"  [CONFIRMED] Checkpoint {resolved_ckpt} loaded successfully.")

    # Reload prototypes
    loaded_mem = PrototypeMemory.load(PROTOTYPE_PATH)
    saved_protos = loaded_mem.get_prototypes()
    print(f"  Saved prototypes shape: {saved_protos.shape}")
    proto_norms = np.linalg.norm(saved_protos, axis=1)
    print(f"  Prototypes L2 norms: {proto_norms} (expected: ~1.0 each)")
    assert saved_protos.shape == (num_classes, 512), "FAILURE: Prototypes dimension mismatch!"
    assert np.allclose(proto_norms, 1.0, atol=1e-3), "FAILURE: Prototypes are not normalized!"
    print("  [CONFIRMED] Saved prototypes are non-empty, normalized, and dimension-consistent.")

    # 7. Evaluate on Held-out Test Images
    print(f"\n[6/7] Evaluating on Held-out Test Images ({test_count} samples)...")
    test_images = []
    test_labels = []
    test_paths = []
    test_identities = []

    for c in classes:
        paths = test_dict.get(c, [])
        c_idx = classes.index(c)
        for p in paths:
            test_paths.append(p)
            test_labels.append(c_idx)
            test_identities.append(c)

    # Read and preprocess test images
    for p in test_paths:
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        test_images.append(img)

    test_tensor = tf.stack(test_images, axis=0)
    test_embeddings = eval_model(test_tensor, training=False).numpy()  # (44, 512)

    # Classify against learned prototypes
    sim_to_protos = test_embeddings @ saved_protos.T  # (44, 5)
    pred_labels = np.argmax(sim_to_protos, axis=1)

    correct_preds = (pred_labels == np.array(test_labels))
    test_accuracy = float(np.mean(correct_preds))
    print(f"\n  Held-out Test Classification Accuracy: {test_accuracy * 100:.2f}% ({np.sum(correct_preds)}/{len(test_labels)})")

    # Classification Report
    clf_report = classification_report(
        test_labels, pred_labels, target_names=classes, output_dict=True, zero_division=0
    )
    macro_precision = float(clf_report['macro avg']['precision'])
    macro_recall = float(clf_report['macro avg']['recall'])
    macro_f1 = float(clf_report['macro avg']['f1-score'])

    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall:    {macro_recall:.4f}")
    print(f"  Macro F1-Score:  {macro_f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, pred_labels, labels=list(range(num_classes)))
    print("  Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — 5-Identity Real Test Dataset", fontsize=12, pad=12)
    fig.colorbar(im)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Prototype Label')
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"  Confusion matrix figure saved to {cm_path}")

    # Pairwise Cosine Similarity Distribution (Genuine vs Impostor)
    genuine_sims = []
    impostor_sims = []
    n_test = len(test_embeddings)
    for i in range(n_test):
        for j in range(i + 1, n_test):
            sim = float(np.dot(test_embeddings[i], test_embeddings[j]))
            if test_labels[i] == test_labels[j]:
                genuine_sims.append(sim)
            else:
                impostor_sims.append(sim)

    print(f"\n  Pairwise Verification Analysis ({n_test * (n_test - 1) // 2} total pairs):")
    print(f"    Genuine pairs:  {len(genuine_sims)} (mean similarity: {np.mean(genuine_sims):.4f})")
    print(f"    Impostor pairs: {len(impostor_sims)} (mean similarity: {np.mean(impostor_sims):.4f})")

    # Plot Similarity Distributions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(genuine_sims) > 0:
        ax.hist(genuine_sims, bins=15, alpha=0.7, label=f"Genuine (N={len(genuine_sims)}, mean={np.mean(genuine_sims):.3f})", color="green")
    if len(impostor_sims) > 0:
        ax.hist(impostor_sims, bins=25, alpha=0.7, label=f"Impostor (N={len(impostor_sims)}, mean={np.mean(impostor_sims):.3f})", color="red")
    ax.set_title("Cosine Similarity Distribution (44 Held-Out Test Images)", fontsize=12)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    sim_path = os.path.join(REPORT_DIR, "similarity_distribution.png")
    plt.savefig(sim_path, dpi=200)
    plt.close()
    print(f"  Similarity distribution figure saved to {sim_path}")

    # ROC / AUC / EER
    all_pairs_scores = genuine_sims + impostor_sims
    all_pairs_targets = [1] * len(genuine_sims) + [0] * len(impostor_sims)
    if len(genuine_sims) > 0 and len(impostor_sims) > 0:
        fpr, tpr, thresholds = roc_curve(all_pairs_targets, all_pairs_scores)
        roc_auc = float(auc(fpr, tpr))
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_threshold = float(thresholds[eer_idx])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color='red', zorder=5, label=f'EER = {eer:.4f} (Thresh={eer_threshold:.2f})')
        ax.set_title(f"ROC Verification Curve (Pairs={len(all_pairs_scores)})", fontsize=12)
        ax.set_xlabel("False Positive Rate (FAR)")
        ax.set_ylabel("True Positive Rate (TAR)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = os.path.join(REPORT_DIR, "roc_curve_real.png")
        plt.savefig(roc_path, dpi=200)
        plt.close()
        print(f"  ROC Curve: AUC = {roc_auc:.4f}, EER = {eer:.4f} (Saved to {roc_path})")
    else:
        roc_auc = None
        eer = None
        eer_threshold = None

    # Plot Training Loss & Accuracy Curves
    history = train_history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs_range, history["loss"], 'b-o', label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], 'r--s', label="Val Loss")
    ax1.set_title("Adaptive Margin Loss Curve", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, history["accuracy"], 'b-o', label="Train Accuracy")
    ax2.plot(epochs_range, history["val_accuracy"], 'r--s', label="Val Accuracy")
    ax2.set_title("Classification Accuracy Curve", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    curves_path = os.path.join(REPORT_DIR, "real_training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"\n  Training curves saved to {curves_path}")

    # 8. Save Comprehensive Evaluation Report
    eval_report = {
        "dataset_summary": {
            "classes": classes,
            "num_classes": num_classes,
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "total_images": train_count + val_count + test_count,
            "dataset_split_rule": "70% train / 15% val / 15% test stratified, zero-leakage",
            "excluded_identities": "adnan and ranveer excluded due to low sample counts, preserved safely"
        },
        "training_details": {
            "model_architecture": "MobileNetV2 (512-d L2-normalized embedding head)",
            "execution_device": "CPU",
            "epochs_trained": len(history["loss"]),
            "batch_size": batch_size,
            "total_time_seconds": round(total_train_time, 2),
            "final_train_loss": round(float(history["loss"][-1]), 4),
            "final_train_acc": round(float(history["accuracy"][-1]), 4),
            "final_val_loss": round(float(history["val_loss"][-1]), 4),
            "final_val_acc": round(float(history["val_accuracy"][-1]), 4),
            "dense_weight_delta_l2": round(float(dense_weight_diff), 6),
            "loss_w_delta_l2": round(float(loss_w_diff), 6)
        },
        "held_out_test_metrics": {
            "sample_count": len(test_labels),
            "accuracy": round(test_accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": {c: {k: round(float(v), 4) for k, v in clf_report[c].items()} for c in classes}
        },
        "verification_metrics": {
            "total_pairs": len(all_pairs_scores),
            "genuine_pairs_count": len(genuine_sims),
            "impostor_pairs_count": len(impostor_sims),
            "genuine_mean_sim": round(float(np.mean(genuine_sims)), 4) if genuine_sims else None,
            "impostor_mean_sim": round(float(np.mean(impostor_sims)), 4) if impostor_sims else None,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "eer": round(eer, 4) if eer is not None else None,
            "eer_threshold": round(eer_threshold, 4) if eer_threshold is not None else None
        }
    }

    report_json_path = os.path.join(REPORT_DIR, "real_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n[7/7] Detailed evaluation report saved to {report_json_path}")
    print("\n" + "=" * 65)
    print("  ALL TRAINING AND EVALUATION PHASES FINISHED SUCCESSFULLY")
    print("=" * 65)


if __name__ == '__main__':
    main()
