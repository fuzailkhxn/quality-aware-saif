"""
src/preprocessing/quality_assessment.py
=========================================
Image quality assessment utilities — supports Objectives C and D.

What is image quality in this context?
---------------------------------------
A face image has high quality if it is:
  - Sharp (not blurry)
  - Adequately lit (not too dark, not overexposed)
  - Appropriately sized (face region is large enough)

How is quality used?
---------------------
1. During prototype updates: high-quality samples get more influence.
2. During loss computation: low-quality samples get a lower loss weight,
   reducing the risk of noisy samples corrupting the model.

Quality metrics implemented
----------------------------
1. Sharpness  — Laplacian variance: high variance = sharp edges = sharp image.
2. Brightness — Mean pixel intensity: too low (dark) or too high (overexposed)
                both reduce quality.
3. Combined   — Weighted product of the two normalised scores.
"""

import numpy as np
import cv2
from typing import Union


def compute_sharpness(
    image: np.ndarray,
    threshold: float = 100.0,
) -> float:
    """
    Measure image sharpness using the variance of the Laplacian operator.

    A high variance means the image contains strong edges → sharp.
    A low variance means smooth / blurry image.

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB uint8 image, shape (H, W, 3) or (H, W).
    threshold : float
        Laplacian variance value considered "perfectly sharp".
        Scores above this are clipped to 1.0.

    Returns
    -------
    float
        Sharpness score in [0, 1]. 0 = very blurry, 1 = very sharp.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(
            (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = min(lap_var / (threshold + 1e-8), 1.0)
    return float(score)


def compute_brightness_score(
    image: np.ndarray,
    low: float = 40.0,
    high: float = 220.0,
) -> float:
    """
    Score how well-lit an image is based on mean pixel intensity.

    Images with mean intensity in the range [low, high] score close to 1.0.
    Too dark (< low) or overexposed (> high) images score closer to 0.

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB uint8 image.
    low : float
        Lower brightness bound (pixels below this → dark image).
    high : float
        Upper brightness bound (pixels above this → overexposed image).

    Returns
    -------
    float
        Brightness score in [0, 1].
    """
    if image.max() <= 1.0:
        pixel_mean = image.mean() * 255.0
    else:
        pixel_mean = float(image.mean())

    if pixel_mean < low:
        # Too dark
        score = pixel_mean / (low + 1e-8)
    elif pixel_mean > high:
        # Overexposed
        score = 1.0 - (pixel_mean - high) / (255.0 - high + 1e-8)
    else:
        # Good brightness range
        score = 1.0

    return float(max(0.0, min(1.0, score)))


def compute_quality_score(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    brightness_low: float = 40.0,
    brightness_high: float = 220.0,
    sharpness_weight: float = 0.6,
    brightness_weight: float = 0.4,
) -> float:
    """
    Compute a combined quality score for a face image.

    Score = sharpness_weight * sharpness_score
           + brightness_weight * brightness_score

    Both sub-scores are in [0, 1], so the combined score is also in [0, 1].

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB image (uint8 or float32 in [0,1]).
    blur_threshold : float
        Laplacian variance considered "sharp". Default 100.
    brightness_low : float
        Minimum acceptable mean pixel intensity. Default 40.
    brightness_high : float
        Maximum acceptable mean pixel intensity. Default 220.
    sharpness_weight : float
        Weight for sharpness component. Default 0.6.
    brightness_weight : float
        Weight for brightness component. Default 0.4.

    Returns
    -------
    float
        Quality score in [0, 1]. Higher is better.
    """
    sharpness = compute_sharpness(image, threshold=blur_threshold)
    brightness = compute_brightness_score(image, low=brightness_low, high=brightness_high)

    combined = sharpness_weight * sharpness + brightness_weight * brightness
    return float(np.clip(combined, 0.0, 1.0))


def batch_quality_scores(
    images: Union[np.ndarray, list],
    blur_threshold: float = 100.0,
    brightness_low: float = 40.0,
    brightness_high: float = 220.0,
) -> np.ndarray:
    """
    Compute quality scores for a batch of images.

    Parameters
    ----------
    images : array-like, shape (N, H, W, 3)
        Batch of images.

    Returns
    -------
    np.ndarray, shape (N,)
        Quality score per image.
    """
    scores = []
    for img in images:
        score = compute_quality_score(
            img,
            blur_threshold=blur_threshold,
            brightness_low=brightness_low,
            brightness_high=brightness_high,
        )
        scores.append(score)
    return np.array(scores, dtype=np.float32)
