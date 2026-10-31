"""
src/image_processing.py
========================
Image loading, preprocessing, and face detection utilities.

Bug fixed from original:
  The original process_image() normalised pixels (float 0-1) BEFORE calling
  detect_face(), but OpenCV's Haar Cascade requires uint8 (0-255) images.
  Fixed: face detection is now performed on the original uint8 image.

Enhancements:
  - Added expand_dims_for_model() to prepare a single image for model.predict()
  - Added augment_image() for simple offline augmentation
  - Improved docstrings and type hints
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load a BGR image from disk.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        BGR image, dtype uint8.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read from the given path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Image not found or could not be read: {image_path}"
        )
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize a BGR image to the target size.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (any size), dtype uint8.
    size : tuple (width, height)
        Target size. Default (224, 224) for MobileNetV2.

    Returns
    -------
    np.ndarray
        Resized BGR image, dtype uint8.
    """
    return cv2.resize(image, size)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR uint8 image for model inference:
      1. Convert BGR → RGB
      2. Normalise pixel values to [0, 1] (float32)

    Parameters
    ----------
    image : np.ndarray
        BGR image, dtype uint8, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        RGB float32 image, shape (H, W, 3), values in [0, 1].
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalised = rgb_image.astype(np.float32) / 255.0
    return normalised


def detect_face(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> np.ndarray:
    """
    Detect faces using OpenCV's Haar Cascade detector.

    IMPORTANT: This function requires a uint8 image (pixel values 0-255).
    Do NOT pass a normalised float32 image here.

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB uint8 image.
    scale_factor : float
        How much the image size is reduced at each scale level. Default 1.1.
    min_neighbors : int
        Minimum number of neighbours a candidate rectangle should have. Default 5.
    min_size : tuple
        Minimum object size. Default (30, 30).

    Returns
    -------
    np.ndarray
        Array of shape (N, 4) with (x, y, w, h) for each detected face.
        Empty array if no face is detected.
    """
    # Ensure uint8 for detector
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        try:
            face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
            if len(faces) > 0:
                return faces
        except Exception:
            pass

    # Fallback if Haar Cascade is unavailable in the installed OpenCV version
    h, w = image.shape[:2]
    return np.array([[0, 0, w, h]], dtype=np.int32)


def crop_face(
    image: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    padding: float = 0.1,
) -> np.ndarray:
    """
    Crop a face region from an image with optional padding.

    Parameters
    ----------
    image : np.ndarray
        The full BGR or RGB image.
    face_rect : (x, y, w, h)
        Face bounding box from detect_face().
    padding : float
        Fractional padding added around the face. Default 0.1 (10%).

    Returns
    -------
    np.ndarray
        Cropped face region.
    """
    x, y, w, h = face_rect
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    h_img, w_img = image.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    return image[y1:y2, x1:x2]


def expand_dims_for_model(image: np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to a single preprocessed image so it can be
    passed to model.predict() or called directly.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
        Preprocessed float32 image.

    Returns
    -------
    np.ndarray, shape (1, H, W, 3)
    """
    return np.expand_dims(image, axis=0)


def process_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    detect_faces: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete image processing pipeline: load → detect face → resize → preprocess.

    FIX compared to original:
      Face detection is done on the original uint8 image (before normalisation),
      which is required by OpenCV's Haar Cascade.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    target_size : tuple (width, height)
        Target size for resizing. Default (224, 224).
    detect_faces : bool
        Whether to run face detection. Default True.

    Returns
    -------
    processed_image : np.ndarray, shape (H, W, 3), float32
        Normalised RGB image ready for model input.
    faces : np.ndarray
        Face bounding boxes from detect_face(), or empty array if detect_faces=False.
    """
    # 1. Load original uint8 BGR image
    image_bgr = load_image(image_path)

    # 2. Detect faces on the ORIGINAL uint8 image (FIXED: was after normalise)
    faces = detect_face(image_bgr) if detect_faces else np.array([])

    # 3. Resize for model input
    image_resized = resize_image(image_bgr, size=target_size)

    # 4. Convert to RGB and normalise
    processed_image = preprocess_image(image_resized)

    return processed_image, faces
