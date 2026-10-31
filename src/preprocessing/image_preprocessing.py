"""
src/preprocessing/image_preprocessing.py
========================================
Image preprocessing pipeline ensuring exact consistency between training and inference:
- MobileNetV2 512-d feature extraction preprocessing (RGB, 224x224, L2 normalization)
- SFace 128-d demo feature extraction preprocessing
- Quality degradation simulators for research stress-testing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_for_mobilenet(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Standardize a face image for MobileNetV2 embedding extraction.

    1. Convert BGR -> RGB
    2. Resize to target_size (default: 224x224)
    3. Rescale pixel intensities to [0, 1] float32

    Parameters
    ----------
    image : np.ndarray
        Face crop, BGR uint8 format.
    target_size : Tuple[int, int]
        (width, height)

    Returns
    -------
    np.ndarray
        RGB float32 image with shape (target_size[1], target_size[0], 3)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input to preprocess_for_mobilenet.")

    # Convert to RGB
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def apply_blur(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Simulate motion or defocus blur using Gaussian filter."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    if k <= 1:
        return image.copy()
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_brightness(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Simulate low lighting or overexposure."""
    return cv2.convertScaleAbs(image, alpha=float(factor), beta=0)


def apply_contrast(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Simulate poor dynamic range or hazy conditions."""
    mean = np.mean(image)
    adjusted = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Simulate ISO sensor noise."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy
