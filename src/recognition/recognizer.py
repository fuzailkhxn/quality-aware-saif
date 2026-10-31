"""
src/recognition/recognizer.py
=============================
Quality-Aware Face Recognizer supporting both Research (MobileNetV2-512)
and Demonstration (OpenCV SFace-128) engines without mixing embedding spaces.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.database import FaceDatabase
from src.preprocessing.face_detection import FaceDetector, crop_face
from src.preprocessing.quality_assessment import FaceQualityAssessor
from src.preprocessing.image_preprocessing import preprocess_for_mobilenet

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFACE_MODEL_PATH = os.path.join(_BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")


class QualityAwareRecognizer:
    """
    Recognizes faces using quality-adaptive decision boundaries and prototype comparison.
    """

    def __init__(
        self,
        engine: str = "MobileNetV2-512",
        database: Optional[FaceDatabase] = None,
        detector: Optional[FaceDetector] = None,
        assessor: Optional[FaceQualityAssessor] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.detector = detector or FaceDetector()
        self.assessor = assessor or FaceQualityAssessor()
        self.weights_path = weights_path

        # Determine dimensionality and appropriate database
        if "sface" in engine.lower():
            self.embedding_dim = 128
            self.engine_name = "SFace-128 (Demo Engine)"
            self.database = database or FaceDatabase(model_name="SFace-128", expected_dim=128)
        else:
            self.embedding_dim = 512
            self.engine_name = "MobileNetV2-512 (Research Engine)"
            self.database = database or FaceDatabase(model_name="MobileNetV2-512", expected_dim=512)

        # Internal model handles
        self._mobilenet_model = None
        self._sface_model = None
        self._init_models()

    def _init_models(self) -> None:
        """Initialize the requested embedding extractor."""
        if "sface" in self.engine.lower():
            if os.path.exists(SFACE_MODEL_PATH) and hasattr(cv2, "FaceRecognizerSF"):
                try:
                    self._sface_model = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
                except Exception as e:
                    print(f"[QualityAwareRecognizer] Warning: Could not initialize SFace: {e}")
        else:
            # Lazy initialize TensorFlow MobileNetV2 embedding model
            try:
                from src.models.face_model import create_embedding_model
                self._mobilenet_model = create_embedding_model(embedding_dim=512)
                # Attempt to load trained weights if present
                candidate_paths = [
                    self.weights_path,
                    os.path.join(_BASE_DIR, "models", "embedding_model.weights.h5"),
                    os.path.join(_BASE_DIR, "models", "face_model.h5"),
                ]
                for p in candidate_paths:
                    if p and os.path.exists(p):
                        try:
                            self._mobilenet_model.load_weights(p, by_name=True, skip_mismatch=True)
                            print(f"[QualityAwareRecognizer] Loaded MobileNetV2 weights from {p}")
                            break
                        except Exception as w_err:
                            pass
            except Exception as e:
                print(f"[QualityAwareRecognizer] Notice: MobileNetV2 initialization message: {e}")

    def extract_embedding(
        self,
        face_crop: np.ndarray,
        raw_detection: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract L2-normalized embedding vector matching current engine.
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Empty face crop passed to extract_embedding.")

        if "sface" in self.engine.lower() and self._sface_model is not None:
            # SFace inference
            aligned = cv2.resize(face_crop, (112, 112))
            feature = self._sface_model.feature(aligned)
            emb = np.asarray(feature, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8)

        # Research Engine: MobileNetV2 512-d
        if self._mobilenet_model is None:
            self._init_models()

        processed = preprocess_for_mobilenet(face_crop, target_size=(224, 224))
        batch = np.expand_dims(processed, axis=0)
        emb = self._mobilenet_model.predict(batch, verbose=0)[0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def recognize(
        self,
        image: np.ndarray,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute full quality-aware face recognition pipeline.
        """
        if image is None or image.size == 0:
            return {
                "face_detected": False,
                "message": "Input image is empty or invalid.",
                "decision": "NO FACE",
                "person": "None",
                "adjusted_confidence": 0.0,
                "quality": None,
                "quality_level": "None",
                "annotated_image": None,
                "face_crop": None,
                "raw_similarity": 0.0,
                "adaptive_threshold": 0.5,
                "all_similarities": {},
                "decision_details": "No image data.",
                "enhancements": [],
            }

        # 1. Face Detection
        primary = self.detector.get_primary_face(image)
        if primary is None:
            return {
                "face_detected": False,
                "message": "No face detected in the image.",
                "decision": "NO FACE",
                "person": "Unknown",
                "adjusted_confidence": 0.0,
                "quality": None,
                "quality_level": "None",
                "annotated_image": image.copy(),
                "face_crop": None,
                "raw_similarity": 0.0,
                "adaptive_threshold": 0.5,
                "all_similarities": {},
                "decision_details": "No faces located with detector.",
                "enhancements": [],
            }

        box = primary["box"]
        crop = crop_face(image, box, padding=0.12)

        # 2. Quality Assessment
        quality_info = self.assessor.assess(crop)
        quality_level = quality_info["quality_level"]
        quality_score = quality_info["overall_score"]
        adaptive_thresh = custom_threshold if custom_threshold is not None else quality_info["adaptive_threshold"]

        # 3. Embedding Extraction
        embedding = self.extract_embedding(crop, raw_detection=primary.get("raw"))

        # 4. Prototype Comparison with Database
        prototypes = self.database.get_all_prototypes()
        similarities = {}
        best_name = "Unknown"
        best_sim = -1.0

        for name, proto in prototypes.items():
            sim = float(np.dot(embedding, proto))
            similarities[name] = max(-1.0, min(1.0, sim))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # 5. Quality-Aware Confidence & Decision Logic
        if best_sim < 0.0:
            best_sim = 0.0

        # Confidence adjusted by image quality: low quality tempers peak confidence
        q_norm = quality_info["overall_score_norm"]
        adjusted_confidence = best_sim * (0.65 + 0.35 * q_norm)
        adjusted_confidence = float(np.clip(adjusted_confidence, 0.0, 1.0))

        # Check against adaptive threshold
        enhancements = []
        if quality_level == "Low":
            enhancements.append("Adaptive Strict Threshold (+0.12) to mitigate noise risk")
        elif quality_level == "Medium":
            enhancements.append("Adaptive Moderate Threshold (+0.04)")
        else:
            enhancements.append("Standard Optimal Margin & Base Threshold")

        if self.database.is_empty():
            decision = "NO IDENTITIES REGISTERED"
            details = "Database is currently empty. Register faces to enable identification."
            recognized_person = "None"
        elif best_sim >= adaptive_thresh:
            decision = "RECOGNIZED"
            recognized_person = best_name
            details = (
                f"Similarity {best_sim:.3f} meets the {quality_level}-quality "
                f"adaptive threshold ({adaptive_thresh:.2f})."
            )
        elif best_sim >= (adaptive_thresh - 0.10) and quality_level == "Low":
            decision = "UNRELIABLE (LOW QUALITY)"
            recognized_person = f"{best_name} (Uncertain)"
            details = (
                f"Candidate {best_name} matched with similarity {best_sim:.3f}, but image quality "
                f"is too low ({quality_score:.1f}/100) to ensure reliable classification."
            )
        else:
            decision = "UNKNOWN PERSON"
            recognized_person = "Unknown"
            details = (
                f"Highest similarity {best_sim:.3f} falls below the required "
                f"threshold ({adaptive_thresh:.2f})."
            )

        # 6. Annotate Image
        annotated = image.copy()
        x, y, w, h = box
        # Box color depends on decision
        color = (40, 180, 40) if decision == "RECOGNIZED" else ((0, 140, 255) if "UNRELIABLE" in decision else (50, 50, 220))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Label
        label = f"{recognized_person} ({adjusted_confidence*100:.1f}%)"
        cv2.putText(annotated, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Landmark points if available
        if primary.get("landmarks") is not None:
            for pt in primary["landmarks"]:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        return {
            "face_detected": True,
            "face_box": box,
            "face_crop": crop,
            "annotated_image": annotated,
            "quality": quality_info,
            "quality_level": quality_level,
            "quality_score": quality_score,
            "embedding": embedding,
            "engine": self.engine_name,
            "person": recognized_person,
            "decision": decision,
            "decision_details": details,
            "raw_similarity": best_sim,
            "adaptive_threshold": adaptive_thresh,
            "adjusted_confidence": adjusted_confidence,
            "all_similarities": similarities,
            "enhancements": enhancements,
        }
