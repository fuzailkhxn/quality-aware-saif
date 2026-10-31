"""
src/database.py
===============
Face Database Module with strict model isolation and quality-aware prototype calculation.

Features:
- Isolates 512-d MobileNetV2 embeddings from 128-d SFace embeddings.
- Computes quality-weighted empirical prototype vectors per identity.
- Persists metadata (timestamp, quality score, image path).
- Supports listing, updating, deleting, and clearing identities.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data")
DEFAULT_MOBILENET_DB = os.path.join(DATA_DIR, "database_mobilenet_512.pkl")
DEFAULT_SFACE_DB = os.path.join(DATA_DIR, "database_sface_128.pkl")


class FaceDatabase:
    """
    Model-specific face database storing embeddings and computing class prototypes.
    """

    def __init__(
        self,
        model_name: str = "MobileNetV2-512",
        expected_dim: int = 512,
        db_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        
        if db_path is not None:
            self.db_path = db_path
        elif "sface" in model_name.lower() or expected_dim == 128:
            self.db_path = DEFAULT_SFACE_DB
        else:
            self.db_path = DEFAULT_MOBILENET_DB

        # Schema: {identity: [{"embedding": np.ndarray, "quality": float, "path": str, ...}]}
        self.records: Dict[str, List[Dict[str, Any]]] = {}
        self.load()

    def load(self) -> None:
        """Load database records from disk if present."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                
                # Check format compatibility
                if isinstance(data, dict):
                    # Check if old format {name: [emb1, emb2]} or new record format
                    converted = {}
                    for name, items in data.items():
                        new_items = []
                        for item in items:
                            if isinstance(item, dict) and "embedding" in item:
                                emb = np.asarray(item["embedding"], dtype=np.float32)
                                if emb.shape[-1] == self.expected_dim:
                                    item["embedding"] = emb
                                    new_items.append(item)
                            elif isinstance(item, (np.ndarray, list)):
                                emb = np.asarray(item, dtype=np.float32)
                                if emb.shape[-1] == self.expected_dim:
                                    new_items.append({
                                        "embedding": emb,
                                        "quality": 1.0,
                                        "path": "",
                                    })
                        if new_items:
                            converted[name] = new_items
                    self.records = converted
            except Exception as e:
                print(f"[FaceDatabase] Notice: Could not load {self.db_path}: {e}")
                self.records = {}
        else:
            self.records = {}

    def save(self) -> None:
        """Save database records to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.records, f)

    def add_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        quality_score: float = 1.0,
        image_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an embedding for an identity with quality weight and metadata.
        """
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Identity name cannot be empty.")

        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.shape[0] != self.expected_dim:
            raise ValueError(
                f"Dimension mismatch for {self.model_name}: "
                f"expected {self.expected_dim}-d, but received {emb.shape[0]}-d vector."
            )

        # Ensure L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        record = {
            "embedding": emb,
            "quality": float(quality_score),
            "path": str(image_path),
            "meta": metadata or {},
        }

        if clean_name not in self.records:
            self.records[clean_name] = []
        self.records[clean_name].append(record)
        self.save()

    def get_identities(self) -> List[str]:
        """Return list of all registered identity names."""
        return sorted(list(self.records.keys()))

    def get_embeddings(self, name: str) -> List[np.ndarray]:
        """Return all raw embedding vectors for an identity."""
        if name not in self.records:
            return []
        return [r["embedding"] for r in self.records[name]]

    def get_prototype(self, name: str) -> Optional[np.ndarray]:
        """
        Calculate quality-weighted empirical prototype vector for an identity:
        P_c = normalize( sum(w_i * f_i) )
        """
        if name not in self.records or not self.records[name]:
            return None

        weighted_sum = np.zeros(self.expected_dim, dtype=np.float32)
        total_weight = 0.0

        for r in self.records[name]:
            w = max(0.05, float(r.get("quality", 1.0)))
            weighted_sum += w * r["embedding"]
            total_weight += w

        if total_weight > 0:
            norm = np.linalg.norm(weighted_sum)
            if norm > 1e-8:
                return weighted_sum / norm
            return weighted_sum / total_weight

        # Fallback to simple mean
        mean_vec = np.mean([r["embedding"] for r in self.records[name]], axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 1e-8 else mean_vec

    def get_all_prototypes(self) -> Dict[str, np.ndarray]:
        """Return mapping of {identity_name: prototype_vector}."""
        prototypes = {}
        for name in self.get_identities():
            p = self.get_prototype(name)
            if p is not None:
                prototypes[name] = p
        return prototypes

    def get_stats(self) -> Dict[str, int]:
        """Return mapping of identity -> number of registered images."""
        return {name: len(items) for name, items in self.records.items()}

    def is_empty(self) -> bool:
        """Check whether database contains registered identities."""
        return len(self.records) == 0

    def delete_identity(self, name: str) -> bool:
        """Remove a single identity."""
        if name in self.records:
            del self.records[name]
            self.save()
            return True
        return False

    def clear(self) -> None:
        """Clear database entirely."""
        self.records = {}
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
