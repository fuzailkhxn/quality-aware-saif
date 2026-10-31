"""
src/models/face_model.py
========================
Provides access to MobileNetV2 embedding and classification model factories.
Compatible with both src.face_model and src.models.face_model import paths.
"""

from src.face_model import create_face_model, create_embedding_model

__all__ = ["create_face_model", "create_embedding_model"]
