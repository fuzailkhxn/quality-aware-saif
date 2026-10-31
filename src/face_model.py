"""
src/face_model.py
=================
Model factory for the Quality-Aware Face Recognition system.

Two models are provided:
  - create_face_model()      : original classifier (preserved for backward compat)
  - create_embedding_model() : new embedding model used by quality-aware training

The embedding model shares the same MobileNetV2 backbone but outputs a
512-dimensional L2-normalised feature vector instead of class probabilities.
Classification is handled externally via prototype comparison or an adaptive
margin loss, giving much more flexibility.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda,
)
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Original model (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def create_face_model(num_classes: int = 2) -> Model:
    """
    Create the original face recognition classifier using MobileNetV2.

    This function is kept unchanged so that existing scripts that import
    create_face_model() continue to work without modification.

    Parameters
    ----------
    num_classes : int
        Number of identity classes. Must be >= 2.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with softmax output.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2 for classification.")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------------------------------------------
# New embedding model for quality-aware training
# ---------------------------------------------------------------------------

def create_embedding_model(
    embedding_dim: int = 512,
    backbone_trainable: bool = False,
    fine_tune_from_layer: int = -30,
    dropout_rate: float = 0.3,
) -> Model:
    """
    Create an embedding model that maps face images to a 512-d (or
    embedding_dim-d) L2-normalised feature vector.

    Architecture
    ------------
    MobileNetV2 backbone (ImageNet weights, top removed)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dropout(dropout_rate)
        -> Dense(embedding_dim, linear activation)   # raw embedding
        -> BatchNormalization                         # stabilise scale
        -> L2-normalise (Lambda layer)               # unit-length vector

    The model does NOT include a classification head. Classification is
    performed externally using prototype memory or an adaptive margin loss,
    making it straightforward to add new identities without retraining the
    full model.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the output embedding vector. Default 512.
    backbone_trainable : bool
        If False (default), the MobileNetV2 backbone weights are frozen
        (standard transfer learning). Set True to fine-tune the full model.
    fine_tune_from_layer : int
        When backbone_trainable is True, only the last abs(fine_tune_from_layer)
        layers are unfrozen. Negative index — e.g. -30 means last 30 layers.
        Ignored when backbone_trainable is False.
    dropout_rate : float
        Dropout probability applied after the global pooling layer.

    Returns
    -------
    tf.keras.Model
        Un-compiled embedding model with output shape (batch_size, embedding_dim).
        The caller is responsible for compiling with an appropriate loss
        (see src/training/losses.py).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    if backbone_trainable:
        # Freeze all layers first, then selectively unfreeze the top ones
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from_layer]:
            layer.trainable = False
    else:
        base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_gap")(x)
    x = Dropout(dropout_rate, name="dropout")(x)

    # Linear projection to embedding dimension
    x = Dense(embedding_dim, activation=None, use_bias=False,
               name="embedding_dense")(x)

    # Batch-normalise for training stability
    x = BatchNormalization(name="bn_embedding")(x)

    # L2-normalise so all embeddings lie on the unit hypersphere.
    # This is required for cosine similarity and ArcFace-style losses.
    embedding = Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="l2_normalize",
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=embedding,
        name="EmbeddingModel",
    )

    return model


def get_backbone_layer_names(embedding_dim: int = 512) -> list:
    """
    Utility — returns the names of all MobileNetV2 backbone layers.
    Useful for deciding which layers to unfreeze during fine-tuning.
    """
    model = create_embedding_model(embedding_dim)
    return [l.name for l in model.layers]
