"""
src/preprocessing/face_detection.py
===================================
Robust face detection module using OpenCV's YuNet ONNX detector with Haar Cascade fallback.
Provides bounding box, 5-point facial landmarks, and detection confidence.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Locate default models directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_YUNET_PATH = os.path.join(_BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")


class FaceDetector:
    """
    Face detector providing bounding boxes, landmarks, and confidence scores.
    Prefers OpenCV YuNet ONNX detector; gracefully falls back to Haar Cascade.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model_path = model_path or DEFAULT_YUNET_PATH

        self._yunet = None
        self._use_yunet = False

        if os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                # YuNet requires initial size; will be updated dynamically in detect()
                self._yunet = cv2.FaceDetectorYN.create(
                    self.model_path,
                    "",
                    (320, 320),
                    self.conf_threshold,
                    self.nms_threshold,
                    self.top_k,
                )
                self._use_yunet = True
            except Exception as e:
                print(f"[FaceDetector] Warning: Could not initialize YuNet: {e}. Falling back to Haar Cascade.")
                self._use_yunet = False

        # Initialize Haar Cascade as fallback
        self._haar_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(cascade_file):
                    self._haar_cascade = cv2.CascadeClassifier(cascade_file)
            except Exception:
                self._haar_cascade = None

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image.

        Parameters
        ----------
        image : np.ndarray
            BGR image, dtype uint8.

        Returns
        -------
        List[Dict[str, Any]]
            List of detected faces, each containing:
            - 'box': (x, y, w, h)
            - 'confidence': float in [0, 1]
            - 'landmarks': array of 5 points (x, y) or None
        """
        if image is None or image.size == 0:
            return []

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        if self._use_yunet and self._yunet is not None:
            try:
                self._yunet.setInputSize((w, h))
                _, raw_faces = self._yunet.detect(image)
                if raw_faces is not None and len(raw_faces) > 0:
                    results = []
                    for face in raw_faces:
                        box = (
                            max(0, int(face[0])),
                            max(0, int(face[1])),
                            min(w - int(face[0]), int(face[2])),
                            min(h - int(face[1]), int(face[3])),
                        )
                        # Landmarks: 5 points: right eye, left eye, nose, right mouth, left mouth
                        landmarks = face[4:14].reshape((5, 2)).astype(np.float32)
                        conf = float(face[-1])
                        results.append({
                            "box": box,
                            "confidence": conf,
                            "landmarks": landmarks,
                            "raw": face,
                        })
                    return results
            except Exception as e:
                # If YuNet fails on this frame, fall back below
                pass

        # Fallback to Haar Cascade
        if self._haar_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            haar_faces = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(haar_faces) > 0:
                results = []
                for (x, y, fw, fh) in haar_faces:
                    results.append({
                        "box": (int(x), int(y), int(fw), int(fh)),
                        "confidence": 0.85,  # Estimated confidence for Haar
                        "landmarks": None,
                        "raw": None,
                    })
                return results

        # Ultimate fallback: treat center crop / entire image as face candidate
        return [{
            "box": (0, 0, w, h),
            "confidence": 0.50,
            "landmarks": None,
            "raw": None,
        }]

    def get_primary_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get the most prominent (highest confidence / largest) face."""
        faces = self.detect_faces(image)
        if not faces:
            return None
        # Sort by area * confidence
        faces.sort(key=lambda f: f["box"][2] * f["box"][3] * f["confidence"], reverse=True)
        return faces[0]


def crop_face(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    padding: float = 0.12,
) -> np.ndarray:
    """
    Crop face with proportional boundary padding.
    """
    x, y, w, h = box
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    h_img, w_img = image.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop
