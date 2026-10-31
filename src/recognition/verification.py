"""
src/recognition/verification.py
===============================
1:1 Face Verification module supporting Objective 5.
Evaluates pair similarity, joint quality score, and adaptive decision boundary.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.recognition.recognizer import QualityAwareRecognizer


def verify_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    recognizer: QualityAwareRecognizer,
    custom_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform 1:1 face verification between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image (BGR).
    image2 : np.ndarray
        Second image (BGR).
    recognizer : QualityAwareRecognizer
        Recognizer instance defining the model engine and detector.
    custom_threshold : Optional[float]
        Optional threshold override.

    Returns
    -------
    Dict[str, Any]
        Verification report including:
        - match: bool
        - similarity: float
        - joint_quality: float
        - adaptive_threshold: float
        - confidence: float
        - decision: str
        - explanation: str
        - crop1, crop2: np.ndarray
        - quality1, quality2: dict
    """
    if image1 is None or image2 is None or image1.size == 0 or image2.size == 0:
        return {
            "valid": False,
            "match": False,
            "error": "One or both images are invalid or empty.",
        }

    # Detect faces
    face1 = recognizer.detector.get_primary_face(image1)
    face2 = recognizer.detector.get_primary_face(image2)

    if face1 is None:
        return {"valid": False, "match": False, "error": "No face detected in Image 1."}
    if face2 is None:
        return {"valid": False, "match": False, "error": "No face detected in Image 2."}

    from src.preprocessing.face_detection import crop_face
    crop1 = crop_face(image1, face1["box"], padding=0.12)
    crop2 = crop_face(image2, face2["box"], padding=0.12)

    # Assess quality
    q1 = recognizer.assessor.assess(crop1)
    q2 = recognizer.assessor.assess(crop2)

    # Joint quality: geometric mean of normalised quality scores
    q_norm1 = q1["overall_score_norm"]
    q_norm2 = q2["overall_score_norm"]
    joint_q_norm = float(np.sqrt(max(0.01, q_norm1 * q_norm2)))
    joint_q_score = round(joint_q_norm * 100.0, 1)

    # Adaptive verification threshold based on joint quality
    if joint_q_score >= 75.0:
        base_thresh = 0.45
        tier = "High"
    elif joint_q_score >= 50.0:
        base_thresh = 0.52
        tier = "Medium"
    else:
        base_thresh = 0.62
        tier = "Low"

    threshold = custom_threshold if custom_threshold is not None else base_thresh

    # Extract embeddings
    emb1 = recognizer.extract_embedding(crop1, raw_detection=face1.get("raw"))
    emb2 = recognizer.extract_embedding(crop2, raw_detection=face2.get("raw"))

    # Cosine similarity
    similarity = float(np.dot(emb1, emb2))
    similarity = max(-1.0, min(1.0, similarity))

    # Quality-tempered confidence
    confidence = similarity * (0.65 + 0.35 * joint_q_norm)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    is_match = similarity >= threshold

    if is_match:
        decision = "VERIFIED MATCH"
        explanation = (
            f"Cosine similarity of {similarity:.3f} meets the {tier}-quality "
            f"adaptive threshold ({threshold:.2f}). Both images represent the SAME identity."
        )
    elif similarity >= (threshold - 0.10) and tier == "Low":
        decision = "UNRELIABLE MATCH"
        explanation = (
            f"Similarity of {similarity:.3f} is close to threshold, but low joint quality "
            f"({joint_q_score}/100) prevents conclusive verification."
        )
    else:
        decision = "DIFFERENT IDENTITIES"
        explanation = (
            f"Cosine similarity of {similarity:.3f} falls below the adaptive "
            f"threshold ({threshold:.2f}). Faces belong to DIFFERENT identities."
        )

    return {
        "valid": True,
        "match": is_match,
        "decision": decision,
        "explanation": explanation,
        "similarity": similarity,
        "joint_quality": joint_q_score,
        "adaptive_threshold": threshold,
        "confidence": confidence,
        "engine": recognizer.engine_name,
        "crop1": crop1,
        "crop2": crop2,
        "quality1": q1,
        "quality2": q2,
    }
