"""
recognize.py
============
Face Recognition and Verification Application.

Features:
1. Verify whether two face images belong to the same person:
       python recognize.py --verify path/to/image1.jpg path/to/image2.jpg

2. Identify who a face image belongs to (matches against learned prototypes):
       python recognize.py --identify path/to/image.jpg

3. Test with the baseline classifier:
       python recognize.py --baseline --identify path/to/image.jpg
"""

import os
import sys
import argparse
import numpy as np

# Ensure root directory is in sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from src.image_processing import load_image, resize_image, preprocess_image, detect_face, crop_face
from src.preprocessing.quality_assessment import compute_quality_score, compute_sharpness, compute_brightness_score
from src.utils.metrics import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Quality-Aware Face Recognition & Verification Application")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--verify",
        nargs=2,
        metavar=("IMAGE1", "IMAGE2"),
        help="Compare two face images to verify if they are the same person",
    )
    group.add_argument(
        "--identify",
        metavar="IMAGE",
        help="Identify a face image against stored identity prototypes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=cfg.VERIFICATION_THRESHOLD,
        help=f"Cosine similarity threshold for verification (default: {cfg.VERIFICATION_THRESHOLD})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.EMBEDDING_WEIGHTS_PATH,
        help=f"Path to embedding model weights or full model (default: {cfg.EMBEDDING_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--prototypes",
        type=str,
        default=cfg.PROTOTYPE_SAVE_PATH,
        help="Path to saved prototypes",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline classifier (models/face_model.h5)",
    )
    return parser.parse_args()


def extract_face_and_quality(image_path: str):
    """Load image, detect face, compute quality score, and return normalized face."""
    bgr = load_image(image_path)
    faces = detect_face(bgr)

    if len(faces) > 0:
        face_rect = faces[0]
        face_crop = crop_face(bgr, face_rect)
        print(f"  [Face Detection] Detected face at {face_rect}")
    else:
        print("  [Face Detection] No frontal face detected with Haar Cascade. Using full image.")
        face_crop = bgr

    # Quality metrics
    sharpness = compute_sharpness(face_crop, threshold=cfg.BLUR_THRESHOLD)
    brightness = compute_brightness_score(face_crop, low=cfg.BRIGHTNESS_LOW, high=cfg.BRIGHTNESS_HIGH)
    quality = compute_quality_score(face_crop)

    # Preprocess for model
    resized = resize_image(face_crop, size=cfg.IMAGE_SIZE)
    normed = preprocess_image(resized)
    input_tensor = np.expand_dims(normed, axis=0)

    return input_tensor, quality, sharpness, brightness


def load_embedding_model_for_inference(model_path: str):
    """
    Load an embedding model for inference.
    Supports Keras 3 weights checkpoints (.weights.h5) and full model files (.keras, .h5).
    """
    import tensorflow as tf
    from src.face_model import create_embedding_model

    candidates = [model_path]
    if model_path.endswith(".h5") and not model_path.endswith(".weights.h5"):
        candidates.append(model_path[:-3] + ".weights.h5")
    elif model_path.endswith(".keras"):
        candidates.append(model_path[:-6] + ".weights.h5")

    for p in [cfg.EMBEDDING_WEIGHTS_PATH, cfg.EMBEDDING_MODEL_PATH, "models/embedding_model.weights.h5"]:
        if p not in candidates:
            candidates.append(p)

    target_path = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            target_path = cand
            break

    if target_path is None:
        raise FileNotFoundError(
            f"Embedding model file '{model_path}' not found.\n"
            f"Searched candidates: {candidates}\n"
            "Please train the model first by running: python train.py"
        )

    # If it is a weights checkpoint file
    if target_path.endswith(".weights.h5") or "_weights" in target_path:
        print(f"  [Model] Loading architecture and restoring weights from: {target_path}")
        model = create_embedding_model(embedding_dim=cfg.EMBEDDING_DIM)
        model.load_weights(target_path)
        return model

    # Otherwise attempt full model load
    try:
        print(f"  [Model] Loading full model from: {target_path}")
        return tf.keras.models.load_model(target_path, compile=False)
    except Exception as e:
        print(f"  [Model] Full model load failed ({e}), attempting to load as weights...")
        model = create_embedding_model(embedding_dim=cfg.EMBEDDING_DIM)
        model.load_weights(target_path)
        return model


def run_verify(img_path1: str, img_path2: str, model_path: str, threshold: float):
    print("\n" + "=" * 60)
    print("  Face Verification (1-to-1 Match)")
    print("=" * 60)

    try:
        model = load_embedding_model_for_inference(model_path)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    print(f"\nProcessing Image 1: {img_path1}")
    face1, q1, s1, b1 = extract_face_and_quality(img_path1)
    print(f"  Quality: {q1:.2f} (Sharpness: {s1:.2f}, Brightness: {b1:.2f})")

    print(f"\nProcessing Image 2: {img_path2}")
    face2, q2, s2, b2 = extract_face_and_quality(img_path2)
    print(f"  Quality: {q2:.2f} (Sharpness: {s2:.2f}, Brightness: {b2:.2f})")

    # Extract 512-D embeddings
    emb1 = model(face1, training=False).numpy()
    emb2 = model(face2, training=False).numpy()

    sim = float(cosine_similarity(emb1, emb2)[0])
    match = sim >= threshold

    print("\n" + "-" * 60)
    print(f"  Cosine Similarity: {sim:.4f}")
    print(f"  Threshold        : {threshold:.4f}")
    print(f"  Result           : {'MATCH (Same Person)' if match else 'NO MATCH (Different People)'}")
    print("=" * 60 + "\n")


def run_identify(image_path: str, model_path: str, proto_path: str, baseline: bool):
    print("\n" + "=" * 60)
    print("  Face Identification")
    print("=" * 60)

    if baseline:
        baseline_path = cfg.MODEL_SAVE_PATH
        if not os.path.exists(baseline_path):
            print(f"\n[Error] Baseline model '{baseline_path}' not found.")
            return

        import tensorflow as tf
        model = tf.keras.models.load_model(baseline_path, compile=False)
        face, q, s, b = extract_face_and_quality(image_path)
        preds = model.predict(face, verbose=0)[0]
        class_id = int(np.argmax(preds))
        conf = float(preds[class_id])

        print(f"\n[Baseline] Predicted Class ID: {class_id} (Confidence: {conf*100:.1f}%)")
        return

    if not os.path.exists(proto_path):
        print(f"\n[Error] Prototypes file '{proto_path}' not found.")
        print("Please train the model first by running: python train.py")
        return

    try:
        model = load_embedding_model_for_inference(model_path)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    from src.models.prototype_memory import PrototypeMemory
    proto_mem = PrototypeMemory.load(proto_path)

    face, q, s, b = extract_face_and_quality(image_path)
    print(f"  Quality: {q:.2f} (Sharpness: {s:.2f}, Brightness: {b:.2f})")

    emb = model(face, training=False).numpy()[0]
    sims = proto_mem.similarity_to_prototypes(emb)
    best_id = int(np.argmax(sims))
    best_score = float(sims[best_id])

    print("\n" + "-" * 60)
    print(f"  Identified Class : Class {best_id}")
    print(f"  Similarity Score : {best_score:.4f}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    if args.verify:
        run_verify(args.verify[0], args.verify[1], args.model, args.threshold)
    elif args.identify:
        run_identify(args.identify, args.model, args.prototypes, args.baseline)


if __name__ == "__main__":
    main()

