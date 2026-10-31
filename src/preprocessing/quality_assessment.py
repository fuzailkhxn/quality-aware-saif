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
  - Good contrast and balanced illumination

How is quality used?
---------------------
1. During prototype updates: high-quality samples get more influence.
2. During loss computation: low-quality samples get a lower loss weight,
   reducing the risk of noisy samples corrupting the model.
3. During inference: dynamically adapts verification and recognition decision thresholds.
"""

import cv2
import numpy as np
from typing import Union, Dict, Any, Tuple


def compute_sharpness(
    image: np.ndarray,
    threshold: float = 100.0,
) -> float:
    """
    Measure image sharpness using the variance of the Laplacian operator.
    Score in [0, 1].
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(
            (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = min(lap_var / (threshold + 1e-8), 1.0)
    return float(max(0.0, score))


def compute_brightness_score(
    image: np.ndarray,
    low: float = 40.0,
    high: float = 220.0,
) -> float:
    """
    Score how well-lit an image is based on mean pixel intensity.
    Score in [0, 1].
    """
    if image is None or image.size == 0:
        return 0.0

    if image.max() <= 1.0:
        pixel_mean = float(image.mean() * 255.0)
    else:
        pixel_mean = float(image.mean())

    if pixel_mean < low:
        score = pixel_mean / (low + 1e-8)
    elif pixel_mean > high:
        score = 1.0 - (pixel_mean - high) / (255.0 - high + 1e-8)
    else:
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
    Compute a combined quality score for a face image in [0, 1].
    """
    if image is None or image.size == 0:
        return 0.0

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
    """Compute quality scores for a batch of images."""
    scores = [
        compute_quality_score(
            img,
            blur_threshold=blur_threshold,
            brightness_low=brightness_low,
            brightness_high=brightness_high,
        )
        for img in images
    ]
    return np.array(scores, dtype=np.float32)


class FaceQualityAssessor:
    """
    Comprehensive Face Quality Assessor for application and research evaluation.
    Evaluates:
    - Sharpness (Laplacian variance)
    - Brightness optimality
    - Dynamic Contrast
    - Face Resolution / Size
    - Illumination Uniformity (left vs right hemisphere)
    """

    def __init__(
        self,
        high_threshold: float = 75.0,
        med_threshold: float = 50.0,
    ) -> None:
        self.high_threshold = high_threshold
        self.med_threshold = med_threshold

    def assess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess an input face crop and return detailed breakdown and categorical tier.
        """
        if image is None or image.size == 0:
            return {
                "overall_score": 0.0,
                "overall_score_norm": 0.0,
                "quality_level": "Low",
                "sharpness": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "face_size": 0.0,
                "illumination": 0.0,
                "details": {
                    "sharpness_label": "Poor",
                    "brightness_label": "Poor",
                    "contrast_label": "Poor",
                    "size_label": "Poor",
                },
                "adaptive_threshold": 0.65,
            }

        # Convert to uint8 BGR if needed
        if image.dtype != np.uint8:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image

        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY) if img_uint8.ndim == 3 else img_uint8

        # 1. Sharpness (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(100.0, max(0.0, (lap_var / 450.0) * 100.0))

        # 2. Brightness (optimal: 100-170)
        mean_b = float(np.mean(gray))
        if 100 <= mean_b <= 170:
            brightness_score = 100.0
        elif mean_b < 100:
            brightness_score = max(0.0, (mean_b / 100.0) * 100.0)
        else:
            brightness_score = max(0.0, ((255.0 - mean_b) / 85.0) * 100.0)

        # 3. Contrast (Standard deviation)
        std_dev = float(np.std(gray))
        contrast_score = min(100.0, max(0.0, (std_dev / 65.0) * 100.0))

        # 4. Face Size / Resolution
        h, w = gray.shape[:2]
        size_min = min(h, w)
        size_score = min(100.0, max(0.0, (size_min / 112.0) * 100.0))

        # 5. Illumination balance (left vs right hemisphere)
        mid_x = w // 2
        left_mean = float(np.mean(gray[:, :mid_x])) if mid_x > 0 else mean_b
        right_mean = float(np.mean(gray[:, mid_x:])) if mid_x > 0 else mean_b
        diff_ratio = abs(left_mean - right_mean) / (max(left_mean, right_mean) + 1e-6)
        illum_score = max(0.0, min(100.0, (1.0 - diff_ratio) * 100.0))

        # Overall weighted composite
        overall = (
            0.35 * sharpness_score +
            0.25 * brightness_score +
            0.15 * contrast_score +
            0.15 * size_score +
            0.10 * illum_score
        )
        overall = round(float(np.clip(overall, 0.0, 100.0)), 1)
        norm_score = float(overall / 100.0)

        if overall >= self.high_threshold:
            quality_level = "High"
            adaptive_threshold = 0.45
        elif overall >= self.med_threshold:
            quality_level = "Medium"
            adaptive_threshold = 0.52
        else:
            quality_level = "Low"
            adaptive_threshold = 0.62

        return {
            "overall_score": overall,
            "overall_score_norm": norm_score,
            "quality_level": quality_level,
            "sharpness": round(sharpness_score, 1),
            "brightness": round(brightness_score, 1),
            "contrast": round(contrast_score, 1),
            "face_size": round(size_score, 1),
            "illumination": round(illum_score, 1),
            "dimensions": (w, h),
            "laplacian_var": round(lap_var, 1),
            "adaptive_threshold": adaptive_threshold,
            "details": {
                "sharpness_label": "Good" if sharpness_score > 60 else ("Fair" if sharpness_score > 30 else "Poor"),
                "brightness_label": "Good" if brightness_score > 60 else ("Fair" if brightness_score > 30 else "Poor"),
                "contrast_label": "Good" if contrast_score > 60 else ("Fair" if contrast_score > 30 else "Poor"),
                "size_label": "Good" if size_score > 60 else ("Fair" if size_score > 30 else "Poor"),
            }
        }
