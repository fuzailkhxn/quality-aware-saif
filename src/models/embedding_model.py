"""
src/models/embedding_model.py
==============================
Thin re-export so that other modules can import the embedding model from
either src.face_model or src.models.embedding_model.

This indirection also allows future backbone swaps (e.g. ResNet50,
EfficientNet) without changing every import site.
"""

from src.face_model import create_embedding_model, create_face_model  # noqa: F401

__all__ = ["create_embedding_model", "create_face_model"]
