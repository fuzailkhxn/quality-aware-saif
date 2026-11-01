"""
src/training/losses.py
=======================
Adaptive Margin Loss — Objective C and D.

Background: ArcFace Loss
-------------------------
Standard ArcFace adds a fixed angular margin m to the angle between a
sample's embedding and its class prototype (weight vector):

    loss = CrossEntropy( s * cos(θ_yi + m),  s * cos(θ_j)  for j ≠ yi )

where s is a scaling factor and θ_yi is the angle to the ground-truth class.

What we add: Adaptive Margins
------------------------------
Instead of a fixed m, we compute a per-sample margin:

    margin_i = clip(m_base + λ · (1 − cos(f_i, P_yi)), m_min, m_max)

where:
  m_base   = base margin (from config, e.g. 0.35 radians)
  λ        = adaptive_lambda (from config, e.g. 0.5)
  cos(f_i, P_yi) = cosine similarity between sample embedding and its
                   prototype vector (how close the sample is to its class center)
  m_min, m_max = clipping bounds for numerical stability

Interpretation:
  - A sample far from its prototype → low cos similarity → large margin
    → the model is pushed harder on difficult samples.
  - A sample close to its prototype → high cos similarity → small margin
    → the model is not over-trained on easy samples.

Noisy Sample Weighting (Objective D):
--------------------------------------
The final loss is a weighted sum:

    L = mean( sample_weight_i * cross_entropy_i )

where:
    sample_weight_i = quality_alpha * quality_score_i
                    + confidence_alpha * confidence_i

  quality_score_i  : sharpness + brightness score [0,1]
  confidence_i     : softmax confidence on correct class [0,1]
  (both from config: QUALITY_WEIGHT_ALPHA, CONFIDENCE_WEIGHT_ALPHA)

Noisy or low-confidence samples get a lower weight, reducing their
gradient contribution WITHOUT completely discarding them.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class AdaptiveMarginLoss(tf.keras.losses.Loss):
    """
    ArcFace loss with per-sample adaptive margins and quality-based weighting.

    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    embedding_dim : int
        Dimensionality of the embedding vectors. Default 512.
    scale : float
        Scaling factor s. Typical value: 64.0.
    base_margin : float
        Base angular margin m_base (radians). Typical value: 0.35.
    adaptive_lambda : float
        Controls how much sample difficulty adjusts the margin. 0 = fixed margin.
    margin_min : float
        Minimum margin (numerical floor).
    margin_max : float
        Maximum margin (prevents extreme margins on very hard samples).
    noise_weight_floor : float
        Minimum loss weight for any sample (so noisy samples still contribute).
    quality_alpha : float
        Weight given to quality score in the sample weight calculation.
    confidence_alpha : float
        Weight given to prediction confidence in the sample weight calculation.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 64.0,
        base_margin: float = 0.35,
        adaptive_lambda: float = 0.5,
        margin_min: float = 0.1,
        margin_max: float = 0.7,
        noise_weight_floor: float = 0.1,
        quality_alpha: float = 0.6,
        confidence_alpha: float = 0.4,
        use_quality_weighting: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.base_margin = base_margin
        self.adaptive_lambda = adaptive_lambda
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.noise_weight_floor = noise_weight_floor
        self.quality_alpha = quality_alpha
        self.confidence_alpha = confidence_alpha
        self.use_quality_weighting = use_quality_weighting

        # Learnable weight matrix W: shape (embedding_dim, num_classes)
        # Each column is the class weight vector (analogous to a prototype in
        # the loss computation; separate from PrototypeMemory which is used
        # for inference and prototype updates).
        self.W = tf.Variable(
            tf.random.normal([embedding_dim, num_classes]),
            trainable=True,
            name="arcface_weights",
            dtype=tf.float32,
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
        quality_scores: Optional[tf.Tensor] = None,
        prototypes: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Callable interface allowing either sample_weight, quality_scores, or prototypes.
        """
        if quality_scores is None and sample_weight is not None:
            quality_scores = sample_weight
        return self.call(y_true, y_pred, quality_scores=quality_scores, prototypes=prototypes)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        quality_scores: Optional[tf.Tensor] = None,
        prototypes: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Compute the adaptive margin loss.

        Parameters
        ----------
        y_true : tf.Tensor, shape (batch, num_classes)
            One-hot encoded ground-truth labels.
        y_pred : tf.Tensor, shape (batch, embedding_dim)
            L2-normalised embedding vectors from the model.
        quality_scores : tf.Tensor, shape (batch,) or None
            Per-sample quality scores in [0,1]. If None, all set to 1.0.
        prototypes : tf.Tensor, shape (num_classes, embedding_dim) or None
            Empirical prototype vectors from PrototypeMemory. When provided,
            sample difficulty is computed from cosine similarity to the
            empirical class prototype cos(f_i, P_yi).

        Returns
        -------
        tf.Tensor, scalar
            Scalar loss value (mean over batch, quality-weighted).
        """
        batch_size = tf.shape(y_pred)[0]

        # 1. Normalise embeddings and weight matrix
        emb_norm = tf.math.l2_normalize(y_pred, axis=1)          # (B, D)
        W_norm   = tf.math.l2_normalize(self.W, axis=0)           # (D, C)

        # 2. Compute cosine similarities between each embedding and all classes
        #    cos_theta[i, j] = cos(angle between sample i and class j)
        cos_theta = tf.matmul(emb_norm, W_norm)                   # (B, C)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # 3. Ground truth labels
        gt_labels = tf.argmax(y_true, axis=1)                     # (B,)
        one_hot = tf.one_hot(gt_labels, self.num_classes)          # (B, C)

        # cos(θ_yi) — similarity to linear classification weight
        cos_gt_linear = tf.reduce_sum(cos_theta * one_hot, axis=1) # (B,)

        # 4. Prototype-Aware Sample Difficulty Calculation (Objective C)
        #    When empirical prototypes are provided, difficulty is directly
        #    measured as distance from the empirical prototype center: (1 - cos(f_i, P_yi))
        if prototypes is not None:
            proto_tensor = tf.cast(prototypes, tf.float32)
            # Check if prototypes are non-zero / initialised
            proto_norms = tf.norm(proto_tensor, axis=1)            # (C,)
            has_valid_protos = tf.reduce_all(proto_norms > 1e-4)

            P_norm = tf.math.l2_normalize(proto_tensor, axis=1)    # (C, D)
            P_gt = tf.gather(P_norm, gt_labels)                     # (B, D)
            cos_proto = tf.reduce_sum(emb_norm * P_gt, axis=1)      # (B,)
            cos_proto = tf.clip_by_value(cos_proto, -1.0 + 1e-7, 1.0 - 1e-7)

            # Use prototype similarity if valid, else fall back to linear weights
            cos_difficulty = tf.where(has_valid_protos, cos_proto, cos_gt_linear)
        else:
            cos_difficulty = cos_gt_linear

        # 5. Adaptive margin per sample
        #    margin_i = clip(m_base + λ · (1 − cos(f_i, P_yi)), m_min, m_max)
        #    Samples far from prototype → larger margin (pushed harder)
        adaptive_margin = self.base_margin + self.adaptive_lambda * (1.0 - cos_difficulty)
        adaptive_margin = tf.clip_by_value(
            adaptive_margin, self.margin_min, self.margin_max
        )                                                           # (B,)

        # 6. Apply margin: add margin to the target-class angle
        theta_gt = tf.acos(cos_gt_linear)                           # (B,)
        theta_margin = theta_gt + adaptive_margin                   # (B,)
        theta_margin = tf.clip_by_value(theta_margin, 0.0, np.pi)

        cos_theta_margin = tf.cos(theta_margin)                     # (B,)

        # 7. Replace the ground-truth class logit with the margined version
        logits = cos_theta * (1.0 - one_hot) + cos_theta_margin[:, None] * one_hot
        logits = logits * self.scale                                # (B, C)

        # 8. Cross-entropy loss per sample (un-reduced)
        per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=logits
        )                                                           # (B,)

        # 8. Quality-based sample weighting (Objective D)
        if not self.use_quality_weighting:
            return tf.reduce_mean(per_sample_loss)

        if quality_scores is None:
            quality_scores = tf.ones(tf.shape(per_sample_loss), dtype=tf.float32)
        else:
            quality_scores = tf.cast(quality_scores, tf.float32)

        # Prediction confidence = softmax probability of the correct class
        probs = tf.nn.softmax(logits, axis=1)                       # (B, C)
        confidence = tf.reduce_sum(probs * one_hot, axis=1)         # (B,)

        # Combined sample weight
        sample_weight = (
            self.quality_alpha     * quality_scores
            + self.confidence_alpha * confidence
        )
        # Ensure weight is never below the floor (noisy samples still contribute)
        sample_weight = tf.maximum(sample_weight, self.noise_weight_floor)
        # Normalise weights to sum to batch size (so effective LR stays stable)
        sample_weight = sample_weight / (
            tf.reduce_mean(sample_weight) + 1e-8
        ) * 1.0

        # 9. Weighted mean loss
        weighted_loss = per_sample_loss * sample_weight
        return tf.reduce_mean(weighted_loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "scale": self.scale,
            "base_margin": self.base_margin,
            "adaptive_lambda": self.adaptive_lambda,
            "margin_min": self.margin_min,
            "margin_max": self.margin_max,
            "noise_weight_floor": self.noise_weight_floor,
            "quality_alpha": self.quality_alpha,
            "confidence_alpha": self.confidence_alpha,
        })
        return config


def build_adaptive_margin_loss(num_classes: int, cfg: object) -> AdaptiveMarginLoss:
    """
    Convenience factory that builds AdaptiveMarginLoss from a config object.

    Parameters
    ----------
    num_classes : int
    cfg : config module (imported config)

    Returns
    -------
    AdaptiveMarginLoss instance
    """
    return AdaptiveMarginLoss(
        num_classes=num_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        scale=cfg.ARC_SCALE,
        base_margin=cfg.ARC_BASE_MARGIN,
        adaptive_lambda=cfg.ADAPTIVE_MARGIN_LAMBDA,
        margin_min=cfg.MARGIN_CLIP_MIN,
        margin_max=cfg.MARGIN_CLIP_MAX,
        noise_weight_floor=cfg.NOISE_WEIGHT_FLOOR,
        quality_alpha=cfg.QUALITY_WEIGHT_ALPHA,
        confidence_alpha=cfg.CONFIDENCE_WEIGHT_ALPHA,
        name="adaptive_margin_loss",
    )
