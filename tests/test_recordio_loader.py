"""
tests/test_recordio_loader.py
==============================
Unit tests for src/preprocessing/recordio_loader.py

These tests cover the pure-Python logic (identity index building, label
remapping, split creation) without needing the actual 2.7 GB train.rec file.

Tests that do require the real dataset (read_record, TF dataset) are
guarded by a skip decorator and only run when CASIA_REC_PATH is set
as an environment variable.

Run from project root:
    .venv\\Scripts\\python.exe -m unittest tests/test_recordio_loader.py -v
"""

import os
import sys
import struct
import tempfile
import unittest
import numpy as np

# Ensure project root is on the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preprocessing.recordio_loader import (
    build_identity_index,
    build_offset_map,
    select_top_n_identities,
    create_recordio_splits,
    create_recordio_train_val_test_splits,
    remap_labels,
    read_record,
    make_recordio_tf_dataset,
    is_recordio_dataset,
)

# -------------------------------------------------------------------
# Path to real CASIA dataset — set this env variable to run live tests
# -------------------------------------------------------------------
_CASIA_DIR = os.environ.get(
    "CASIA_REC_PATH",
    r"C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112"
)
_HAS_REAL_DATA = (
    os.path.isfile(os.path.join(_CASIA_DIR, "train.rec")) and
    os.path.isfile(os.path.join(_CASIA_DIR, "train.idx")) and
    os.path.isfile(os.path.join(_CASIA_DIR, "train.lst"))
)


# ===================================================================
# Helper: create a minimal synthetic LST file
# ===================================================================

def _make_synthetic_lst(tmp_dir: str, records: list) -> str:
    """
    records : list of (record_id, label)  (0-based line numbers will be 1-based)

    Writes a file with format:  0\t/path/img.jpg\t<label>\t0\t0\t112\t112
    Returns path to LST file.
    """
    lst_path = os.path.join(tmp_dir, "train.lst")
    with open(lst_path, "w") as fh:
        for rid, label in records:
            fh.write(f"0\t/data/img_{rid}.jpg\t{label}\t0\t0\t112\t112\n")
    return lst_path


def _make_synthetic_idx(tmp_dir: str, offsets: dict) -> str:
    """offsets: {record_id: byte_offset}"""
    idx_path = os.path.join(tmp_dir, "train.idx")
    with open(idx_path, "w") as fh:
        for rid, off in sorted(offsets.items()):
            fh.write(f"{rid}\t{off}\n")
    return idx_path


# ===================================================================
# Test 1: build_identity_index — basic correctness
# ===================================================================

class TestBuildIdentityIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_correct_label_grouping(self):
        """Records are correctly grouped by their identity label."""
        # 5 records: labels 0, 1, 0, 2, 1
        records = [(1, 0), (2, 1), (3, 0), (4, 2), (5, 1)]
        lst_path = _make_synthetic_lst(self.tmp, records)
        index = build_identity_index(lst_path)

        self.assertIn(0, index)
        self.assertIn(1, index)
        self.assertIn(2, index)
        self.assertEqual(sorted(index[0]), [1, 3])
        self.assertEqual(sorted(index[1]), [2, 5])
        self.assertEqual(sorted(index[2]), [4])

    def test_empty_file(self):
        """An empty LST file returns an empty index (no crash)."""
        lst_path = os.path.join(self.tmp, "empty.lst")
        with open(lst_path, "w") as fh:
            fh.write("")
        index = build_identity_index(lst_path)
        self.assertEqual(len(index), 0)

    def test_malformed_lines_skipped(self):
        """Lines with fewer than 3 tab-separated fields are skipped."""
        lst_path = os.path.join(self.tmp, "bad.lst")
        with open(lst_path, "w") as fh:
            fh.write("0\tbad_line_only_two_fields\n")
            fh.write("0\t/img.jpg\t5\t0\t0\t112\t112\n")
        index = build_identity_index(lst_path)
        # Only the second line is valid
        self.assertEqual(list(index.keys()), [5])


# ===================================================================
# Test 2: build_offset_map
# ===================================================================

class TestBuildOffsetMap(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_correct_offset_parsing(self):
        """Record IDs and byte offsets are parsed correctly."""
        offsets = {1: 0, 2: 1024, 3: 4096}
        idx_path = _make_synthetic_idx(self.tmp, offsets)
        offset_map = build_offset_map(idx_path)

        self.assertEqual(offset_map[1], 0)
        self.assertEqual(offset_map[2], 1024)
        self.assertEqual(offset_map[3], 4096)
        self.assertEqual(len(offset_map), 3)


# ===================================================================
# Test 3: select_top_n_identities
# ===================================================================

class TestSelectTopNIdentities(unittest.TestCase):

    def _make_index(self, counts: dict) -> dict:
        """Make {label: [record_ids]} with the given image counts."""
        return {
            label: list(range(count))
            for label, count in counts.items()
        }

    def test_selects_most_populated_identities(self):
        """Top-N selection picks identities with the most images."""
        counts = {0: 5, 1: 50, 2: 30, 3: 20, 4: 10}
        index = self._make_index(counts)
        selected = select_top_n_identities(index, n=3, min_images=1)

        # Should pick labels 1 (50), 2 (30), 3 (20)
        self.assertEqual(set(selected.keys()), {1, 2, 3})

    def test_min_images_filter(self):
        """Identities with fewer images than min_images are excluded."""
        counts = {0: 5, 1: 50, 2: 3}   # label 2 has only 3 images
        index = self._make_index(counts)
        selected = select_top_n_identities(index, n=10, min_images=5)

        self.assertNotIn(2, selected)
        self.assertIn(0, selected)
        self.assertIn(1, selected)

    def test_n_larger_than_available(self):
        """Requesting more identities than available returns all valid ones."""
        counts = {0: 10, 1: 20}
        index = self._make_index(counts)
        selected = select_top_n_identities(index, n=100, min_images=5)
        self.assertEqual(set(selected.keys()), {0, 1})


# ===================================================================
# Test 4: create_recordio_splits — identity-disjoint guarantee
# ===================================================================

class TestCreateRecordioSplits(unittest.TestCase):

    def _make_index(self, n_identities: int, images_each: int = 15) -> dict:
        return {
            i: list(range(i * images_each, (i + 1) * images_each))
            for i in range(n_identities)
        }

    def test_no_label_overlap_between_splits(self):
        """No identity label appears in more than one split."""
        index = self._make_index(50)
        train, val, test = create_recordio_splits(index, val_fraction=0.15, test_fraction=0.15)

        train_ids = set(train.keys())
        val_ids   = set(val.keys())
        test_ids  = set(test.keys())

        self.assertEqual(len(train_ids & val_ids), 0,  "train ∩ val must be empty")
        self.assertEqual(len(train_ids & test_ids), 0, "train ∩ test must be empty")
        self.assertEqual(len(val_ids & test_ids), 0,   "val ∩ test must be empty")

    def test_all_identities_accounted_for(self):
        """Every identity ends up in exactly one split."""
        index = self._make_index(50)
        train, val, test = create_recordio_splits(index, val_fraction=0.15, test_fraction=0.15)

        all_ids = set(train.keys()) | set(val.keys()) | set(test.keys())
        self.assertEqual(all_ids, set(index.keys()))

    def test_train_has_most_identities(self):
        """Training split is the largest (fraction-wise)."""
        index = self._make_index(100)
        train, val, test = create_recordio_splits(index, val_fraction=0.15, test_fraction=0.15)
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(train), len(test))

    def test_reproducibility(self):
        """Same seed produces identical splits."""
        index = self._make_index(40)
        t1, v1, te1 = create_recordio_splits(index, seed=7)
        t2, v2, te2 = create_recordio_splits(index, seed=7)
        self.assertEqual(set(t1.keys()), set(t2.keys()))
        self.assertEqual(set(v1.keys()), set(v2.keys()))


# ===================================================================
# Test 5: remap_labels
# ===================================================================

class TestRemapLabels(unittest.TestCase):

    def test_contiguous_zero_based_labels(self):
        """Remapped labels form a contiguous 0…N-1 range."""
        index = {100: [1, 2], 500: [3], 999: [4, 5, 6]}
        remapped, label_map = remap_labels(index)

        self.assertEqual(sorted(remapped.keys()), [0, 1, 2])
        # label_map: original → new
        self.assertEqual(label_map[100], 0)
        self.assertEqual(label_map[500], 1)
        self.assertEqual(label_map[999], 2)

    def test_record_ids_preserved(self):
        """Record IDs are unchanged after remapping."""
        index = {200: [10, 20], 300: [30]}
        remapped, _ = remap_labels(index)
        all_rids = set()
        for rids in remapped.values():
            all_rids.update(rids)
        self.assertEqual(all_rids, {10, 20, 30})


# ===================================================================
# Test 6: read_record — with a synthetic .rec file
# ===================================================================

class TestReadRecord(unittest.TestCase):
    """
    Creates a minimal valid RecordIO record in memory and tests read_record().
    Uses a tiny 10×10 JPEG so the test runs quickly without any dataset.
    """

    def _make_synthetic_rec(self, tmp_dir: str):
        """
        Build a synthetic train.rec containing one valid record with a 10×10 JPEG.
        Returns (rec_path, offset_map, identity_label).
        """
        from PIL import Image as PILImage
        from io import BytesIO

        # Create a 10×10 red JPEG
        img = PILImage.new("RGB", (10, 10), color=(220, 50, 50))
        jpeg_buf = BytesIO()
        img.save(jpeg_buf, format="JPEG")
        jpeg_bytes = jpeg_buf.getvalue()

        identity_label = 42

        # Build payload: 24-byte header + JPEG
        # Header: flag(int32)=0 + label(float32)=42.0 + 16 bytes padding
        header = struct.pack("<if", 0, float(identity_label)) + b"\x00" * 16
        assert len(header) == 24, f"Header must be 24 bytes, got {len(header)}"
        payload = header + jpeg_bytes

        # Build record: magic(uint32) + length(uint32) + payload
        magic = 0xCED7230A
        rec_bytes = struct.pack("<II", magic, len(payload)) + payload

        rec_path = os.path.join(tmp_dir, "synth.rec")
        with open(rec_path, "wb") as fh:
            fh.write(rec_bytes)

        # record_id=1 → offset=0 (starts at byte 0)
        offset_map = {1: 0}
        return rec_path, offset_map, identity_label

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_read_record_returns_correct_shape(self):
        """read_record() returns (image_array, label) with correct shape."""
        rec_path, offset_map, expected_label = self._make_synthetic_rec(self.tmp)
        target_size = (32, 32)
        with open(rec_path, "rb") as rf:
            image, label = read_record(rf, offset_map, record_id=1, target_size=target_size)

        self.assertEqual(image.shape, (32, 32, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(0.0 <= image.min() <= image.max() <= 1.0)
        self.assertEqual(label, expected_label)

    def test_missing_record_id_raises_key_error(self):
        """KeyError is raised when record_id is not in offset_map."""
        rec_path, offset_map, _ = self._make_synthetic_rec(self.tmp)
        with open(rec_path, "rb") as rf:
            with self.assertRaises(KeyError):
                read_record(rf, offset_map, record_id=999)

    def test_bad_magic_raises_value_error(self):
        """ValueError is raised when the magic number is wrong (corrupted record)."""
        rec_path = os.path.join(self.tmp, "bad_magic.rec")
        # Write a record with wrong magic (0xDEADBEEF instead of 0xCED7230A)
        payload = b"\x00" * 24 + b"\xff\xd8\xff\xe0" + b"\x00" * 100
        rec_bytes = struct.pack("<II", 0xDEADBEEF, len(payload)) + payload
        with open(rec_path, "wb") as fh:
            fh.write(rec_bytes)

        offset_map = {1: 0}
        with open(rec_path, "rb") as rf:
            with self.assertRaises(ValueError):
                read_record(rf, offset_map, record_id=1)


# ===================================================================
# Test 7: live dataset tests (skipped if dataset not available)
# ===================================================================

@unittest.skipUnless(_HAS_REAL_DATA, "Real CASIA dataset not found — skipping live tests")
class TestLiveDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rec_path = os.path.join(_CASIA_DIR, "train.rec")
        cls.idx_path = os.path.join(_CASIA_DIR, "train.idx")
        cls.lst_path = os.path.join(_CASIA_DIR, "train.lst")

        cls.identity_index = build_identity_index(cls.lst_path)
        cls.offset_map     = build_offset_map(cls.idx_path)

    def test_identity_count(self):
        """CASIA-WebFace has 10,572 identities."""
        self.assertEqual(len(self.identity_index), 10572)

    def test_total_record_count(self):
        """Total record count is approximately 494,149."""
        total = sum(len(v) for v in self.identity_index.values())
        # Allow ±1000 for edge cases
        self.assertAlmostEqual(total, 494149, delta=1000)

    def test_read_first_record_shape(self):
        """First record decodes to a valid (112, 112, 3) float32 image."""
        # Find a record_id that exists in both lst and idx
        first_label = sorted(self.identity_index.keys())[0]
        first_rid = self.identity_index[first_label][0]
        if first_rid not in self.offset_map:
            self.skipTest("First record ID not in offset map")

        with open(self.rec_path, "rb") as rf:
            image, label = read_record(
                rf, self.offset_map, first_rid, target_size=(112, 112)
            )

        self.assertEqual(image.shape, (112, 112, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(0.0 <= image.min() <= image.max() <= 1.0)

    def test_top_100_selection(self):
        """Top-100 selection returns exactly 100 identities."""
        selected = select_top_n_identities(
            self.identity_index, n=100, min_images=10
        )
        self.assertEqual(len(selected), 100)

    def test_splits_are_identity_disjoint(self):
        """Train / val / test splits share no identity labels."""
        selected = select_top_n_identities(
            self.identity_index, n=50, min_images=10
        )
        train, val, test = create_recordio_splits(selected, seed=42)
        train_ids = set(train.keys())
        val_ids   = set(val.keys())
        test_ids  = set(test.keys())

        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)


# ===================================================================
# Test 8: is_recordio_dataset detection
# ===================================================================

class TestRecordIODetection(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_detects_recordio_when_rec_and_idx_exist(self):
        """Returns True when train.rec and train.idx are both present."""
        open(os.path.join(self.tmp, "train.rec"), "w").close()
        open(os.path.join(self.tmp, "train.idx"), "w").close()
        self.assertTrue(is_recordio_dataset(self.tmp))

    def test_detects_false_when_missing_rec(self):
        """Returns False when train.rec is missing."""
        open(os.path.join(self.tmp, "train.idx"), "w").close()
        self.assertFalse(is_recordio_dataset(self.tmp))

    def test_detects_false_when_folder_empty_or_subfolders(self):
        """Returns False for standard directory without RecordIO files."""
        sub = os.path.join(self.tmp, "person1")
        os.makedirs(sub, exist_ok=True)
        self.assertFalse(is_recordio_dataset(self.tmp))

    def test_detects_false_on_nonexistent_directory(self):
        """Returns False safely for nonexistent path."""
        self.assertFalse(is_recordio_dataset(os.path.join(self.tmp, "nonexistent")))


# ===================================================================
# Test 9: create_recordio_train_val_test_splits (3-way partition)
# ===================================================================

class TestTrainValTestSplits(unittest.TestCase):

    def _make_index(self, n_identities: int, images_each: int = 20) -> dict:
        return {
            i: list(range(i * images_each, (i + 1) * images_each))
            for i in range(n_identities)
        }

    def test_matching_classes_between_train_and_val(self):
        """Train and val splits contain the same class keys for loss computation."""
        index = self._make_index(20, images_each=20)
        train, val, test = create_recordio_train_val_test_splits(
            index, val_image_fraction=0.2, test_identity_fraction=0.2, seed=42
        )
        # Train and val have the exact same identities
        self.assertEqual(set(train.keys()), set(val.keys()))
        # Test identities are completely disjoint from train and val
        self.assertEqual(len(set(train.keys()) & set(test.keys())), 0)
        self.assertEqual(len(set(val.keys()) & set(test.keys())), 0)

    def test_disjoint_images_between_train_and_val(self):
        """Images in train and val do not overlap for any identity."""
        index = self._make_index(10, images_each=30)
        train, val, test = create_recordio_train_val_test_splits(
            index, val_image_fraction=0.2, test_identity_fraction=0.2, seed=42
        )
        for identity in train:
            train_rids = set(train[identity])
            val_rids = set(val[identity])
            self.assertEqual(len(train_rids & val_rids), 0, f"Overlap in images for id {identity}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

