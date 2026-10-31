"""
tests/test_dataset_loader.py
=============================
Unit tests for the CASIA-WebFace and identity dataset loader.

Can be executed with:
    python -m unittest tests/test_dataset_loader.py
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.dataset_loader import (
    is_valid_image,
    scan_and_validate_dataset,
    create_identity_disjoint_splits,
    create_closed_set_splits,
    build_tf_dataset_from_dict,
)


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_dummy_image(self, path: str, size=(64, 64)):
        arr = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(path, format="JPEG")

    def test_is_valid_image(self):
        valid_path = os.path.join(self.root_path, "good.jpg")
        self._create_dummy_image(valid_path)
        self.assertTrue(is_valid_image(valid_path))

        # Test empty / 0-byte file
        corrupt_path = os.path.join(self.root_path, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"")
        self.assertFalse(is_valid_image(corrupt_path))

        # Test non-existent file
        self.assertFalse(is_valid_image(os.path.join(self.root_path, "missing.jpg")))

    def test_scan_and_filter_identities(self):
        # Create 3 identities: id1 has 5 images, id2 has 12 images, id3 has 15 images
        counts = {"id1": 5, "id2": 12, "id3": 15}
        for identity, count in counts.items():
            id_dir = os.path.join(self.root_path, identity)
            os.makedirs(id_dir, exist_ok=True)
            for i in range(count):
                self._create_dummy_image(os.path.join(id_dir, f"img_{i}.jpg"))

        # Min images = 10 should filter out id1
        dataset_dict = scan_and_validate_dataset(
            self.root_path,
            min_images_per_identity=10,
            max_identities=None,
        )
        self.assertIn("id2", dataset_dict)
        self.assertIn("id3", dataset_dict)
        self.assertNotIn("id1", dataset_dict)
        self.assertEqual(len(dataset_dict["id2"]), 12)
        self.assertEqual(len(dataset_dict["id3"]), 15)

    def test_max_identities_limit(self):
        for i in range(5):
            id_dir = os.path.join(self.root_path, f"person_{i}")
            os.makedirs(id_dir, exist_ok=True)
            for j in range(10):
                self._create_dummy_image(os.path.join(id_dir, f"img_{j}.jpg"))

        dataset_dict = scan_and_validate_dataset(
            self.root_path,
            min_images_per_identity=5,
            max_identities=3,
        )
        self.assertEqual(len(dataset_dict), 3)

    def test_identity_disjoint_split_has_zero_leakage(self):
        # Create dataset dict with 4 identities
        dataset_dict = {
            f"subject_{i}": [f"/path/{i}_{j}.jpg" for j in range(10)]
            for i in range(4)
        }
        train_dict, val_dict = create_identity_disjoint_splits(dataset_dict, val_fraction=0.25)

        train_ids = set(train_dict.keys())
        val_ids = set(val_dict.keys())

        # Disjoint: no shared identities
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertEqual(len(train_ids) + len(val_ids), 4)
        self.assertGreaterEqual(len(train_ids), 2)
        self.assertGreaterEqual(len(val_ids), 1)

    def test_build_tf_dataset(self):
        # Create 2 identities with 4 images each
        dataset_dict = {}
        for identity in ["person_A", "person_B"]:
            id_dir = os.path.join(self.root_path, identity)
            os.makedirs(id_dir, exist_ok=True)
            paths = []
            for j in range(4):
                p = os.path.join(id_dir, f"img_{j}.jpg")
                self._create_dummy_image(p)
                paths.append(p)
            dataset_dict[identity] = paths

        ds, class_names, num_classes = build_tf_dataset_from_dict(
            dataset_dict,
            image_size=(32, 32),
            batch_size=2,
            augment=False,
            is_training=False,
        )

        self.assertEqual(num_classes, 2)
        self.assertEqual(class_names, ["person_A", "person_B"])

        # Fetch one batch
        for images, labels in ds.take(1):
            self.assertEqual(images.shape, (2, 32, 32, 3))
            self.assertEqual(labels.shape, (2, 2))
            break


if __name__ == "__main__":
    unittest.main()

