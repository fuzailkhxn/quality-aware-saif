"""
recordio_loader.py
==================
Reads MXNet RecordIO files (train.rec + train.idx) from the CASIA-WebFace
dataset directly — no image extraction to disk required.

Plain-English description
--------------------------
MXNet RecordIO stores many images packed end-to-end in one big binary file
(train.rec).  A companion index file (train.idx) tells us the exact byte
position of each image inside train.rec.  A list file (train.lst) maps
each image to its identity (person) label.

This module:
  1. Reads train.lst  → builds {identity_label: [record_ids]}
  2. Reads train.idx  → builds {record_id: byte_offset}
  3. Provides read_record() to decode one JPEG image from train.rec
  4. Selects up to N identities (with most images)
  5. Creates identity-disjoint train / validation / test splits
  6. Builds a tf.data.Dataset (lazy — one seek per image, no full extraction)

RecordIO binary layout (each record)
--------------------------------------
  Bytes 0-3   : magic number  0xCED7230A  (4 bytes, little-endian uint32)
  Bytes 4-7   : record length in bytes    (4 bytes, little-endian uint32)
  Payload     : <length> bytes
    Bytes 0-3   (of payload): flag   (int32, always 0 for image records)
    Bytes 4-7   (of payload): label  (float32, identity label)
    Bytes 8-23  (of payload): padding / secondary fields (16 bytes total header)
    Bytes 24+   (of payload): raw JPEG bytes  ← confirmed by probing ffd8ffe0
"""

import os
import struct
import math
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

# PIL is used only for JPEG decode (no opencv dependency here).
from PIL import Image

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
_MAGIC = 0xCED7230A          # RecordIO magic number (each record starts with this)
_HEADER_BYTES = 24           # Bytes before JPEG in payload (confirmed by binary probe)
_RECORD_HDR_STRUCT = "<II"   # little-endian: uint32 magic + uint32 length
_RECORD_HDR_SIZE = 8         # struct.calcsize("<II") == 8


# ===================================================================
# Dataset type detection helper
# ===================================================================

def is_recordio_dataset(dataset_path: str) -> bool:
    """
    Check if a directory contains MXNet RecordIO files (train.rec and train.idx).

    Parameters
    ----------
    dataset_path : str
        Directory to inspect.

    Returns
    -------
    bool
        True if train.rec and train.idx both exist in dataset_path, False otherwise.
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        return False
    rec_path = os.path.join(dataset_path, "train.rec")
    idx_path = os.path.join(dataset_path, "train.idx")
    return os.path.isfile(rec_path) and os.path.isfile(idx_path)


# ===================================================================
# Step 1 – Parse train.lst
# ===================================================================

def build_identity_index(lst_path: str) -> Dict[int, List[int]]:
    """
    Parse train.lst and return a mapping  {identity_label: [record_id, ...]}.

    train.lst format (tab-separated):
        <unused>  <image_path>  <identity_label>  [bbox + landmark coords ...]

    The first column is unused (often 0).
    The third column (index 2) is the integer identity label (0–10571).

    record_id is the 1-based line number in train.lst (matches train.idx row).

    Returns
    -------
    identity_index : dict
        Keys   : integer identity labels  (0 … 10571)
        Values : list of 1-based record IDs belonging to that identity
    """
    identity_index: Dict[int, List[int]] = {}

    print(f"[RecordIO] Parsing identity index from: {lst_path}")
    with open(lst_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue  # malformed line – skip
            try:
                label = int(float(parts[2]))
            except (ValueError, IndexError):
                continue  # skip header or bad line
            if label not in identity_index:
                identity_index[label] = []
            identity_index[label].append(line_no)  # 1-based record id

    total_records = sum(len(v) for v in identity_index.values())
    print(f"[RecordIO] Found {len(identity_index):,} identities, "
          f"{total_records:,} records in LST")
    return identity_index


# ===================================================================
# Step 2 – Parse train.idx
# ===================================================================

def build_offset_map(idx_path: str) -> Dict[int, int]:
    """
    Parse train.idx and return {record_id: byte_offset_in_rec}.

    train.idx format (tab-separated):
        <record_id>  <byte_offset>

    record_id is 1-based; byte_offset is the position in train.rec
    where that record begins (magic number included).

    Returns
    -------
    offset_map : dict
        {record_id (int): byte_offset (int)}
    """
    offset_map: Dict[int, int] = {}

    print(f"[RecordIO] Parsing byte-offset index from: {idx_path}")
    with open(idx_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                record_id = int(parts[0])
                offset = int(parts[1])
                offset_map[record_id] = offset
            except ValueError:
                continue

    print(f"[RecordIO] Offset map contains {len(offset_map):,} entries")
    return offset_map


# ===================================================================
# Step 3 – Read one record from train.rec
# ===================================================================

def read_record(
    rec_file,           # open binary file handle (already opened)
    offset_map: Dict[int, int],
    record_id: int,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, int]:
    """
    Read and decode one image from an open train.rec file handle.

    Parameters
    ----------
    rec_file    : open binary file (mode='rb')
    offset_map  : {record_id: byte_offset} (from build_offset_map)
    record_id   : which record to read (1-based)
    target_size : (height, width) to resize the decoded image to

    Returns
    -------
    image_array : np.ndarray  shape=(target_size[0], target_size[1], 3)
                  dtype=float32, values in [0, 1]
    label       : int   identity label (0–10571)

    Raises
    ------
    KeyError   : if record_id is not in offset_map
    ValueError : if magic bytes do not match (corrupted record)
    """
    offset = offset_map[record_id]  # raises KeyError if missing

    # Seek to this record's start position in the .rec file
    rec_file.seek(offset)

    # Read the 8-byte record header (magic + length)
    hdr_bytes = rec_file.read(_RECORD_HDR_SIZE)
    if len(hdr_bytes) < _RECORD_HDR_SIZE:
        raise ValueError(f"Truncated record header at record_id={record_id}")

    magic, length = struct.unpack(_RECORD_HDR_STRUCT, hdr_bytes)
    if magic != _MAGIC:
        raise ValueError(
            f"Bad magic at record_id={record_id}: "
            f"expected 0x{_MAGIC:08X}, got 0x{magic:08X}"
        )

    # Read the full payload
    payload = rec_file.read(length)
    if len(payload) < _HEADER_BYTES + 2:
        raise ValueError(f"Payload too short at record_id={record_id}")

    # Extract label from payload bytes 4–7 (float32 in MXNet format)
    label = int(struct.unpack("<f", payload[4:8])[0])

    # JPEG data starts at byte offset 24 inside the payload
    jpeg_bytes = payload[_HEADER_BYTES:]

    # Decode JPEG → numpy
    img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    image_array = np.array(img, dtype=np.float32) / 255.0

    return image_array, label


# ===================================================================
# Step 4 – Identity selection
# ===================================================================

def select_top_n_identities(
    identity_index: Dict[int, List[int]],
    n: int = 100,
    min_images: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Select up to `n` identities that have at least `min_images` records.

    Strategy: pick identities with the MOST images first (most informative
    for training), then truncate to `n`.

    Returns a NEW dict (subset of identity_index).
    """
    # Filter by minimum image count
    filtered = {
        label: rids
        for label, rids in identity_index.items()
        if len(rids) >= min_images
    }

    print(f"[RecordIO] Identities with >= {min_images} images: {len(filtered):,}")

    # Sort descending by image count
    sorted_labels = sorted(filtered.keys(), key=lambda l: -len(filtered[l]))

    # Select top n
    selected_labels = sorted_labels[:n]
    selected = {label: filtered[label] for label in selected_labels}

    total_imgs = sum(len(v) for v in selected.values())
    print(f"[RecordIO] Selected {len(selected)} identities, {total_imgs:,} images total")
    return selected


# ===================================================================
# Step 5 – Identity-disjoint splits
# ===================================================================

def create_recordio_splits(
    identity_index: Dict[int, List[int]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Split identities into train / validation / test sets.

    KEY GUARANTEE: No identity appears in more than one split.
    This is called an "identity-disjoint" split — it prevents data leakage
    where the model sees the same person during training and evaluation.

    Parameters
    ----------
    identity_index : {label: [record_ids]}
    val_fraction   : fraction of identities to hold out for validation
    test_fraction  : fraction of identities to hold out for testing
    seed           : reproducibility

    Returns
    -------
    train_index, val_index, test_index : three dicts (same structure)
    """
    rng = np.random.default_rng(seed)

    all_labels = np.array(sorted(identity_index.keys()))
    rng.shuffle(all_labels)

    n = len(all_labels)
    n_test = max(1, int(math.ceil(n * test_fraction)))
    n_val  = max(1, int(math.ceil(n * val_fraction)))
    n_train = n - n_test - n_val

    if n_train < 2:
        raise ValueError(
            f"Too few identities ({n}) for requested split fractions. "
            f"Reduce val_fraction or test_fraction, or select more identities."
        )

    train_labels = all_labels[:n_train]
    val_labels   = all_labels[n_train: n_train + n_val]
    test_labels  = all_labels[n_train + n_val:]

    train_index = {int(l): identity_index[int(l)] for l in train_labels}
    val_index   = {int(l): identity_index[int(l)] for l in val_labels}
    test_index  = {int(l): identity_index[int(l)] for l in test_labels}

    print(
        f"[RecordIO] Split -> "
        f"train={len(train_index)} ids, "
        f"val={len(val_index)} ids, "
        f"test={len(test_index)} ids"
    )
    return train_index, val_index, test_index


# ===================================================================
# Step 6 – Remap identity labels to 0-based contiguous integers
# ===================================================================

def remap_labels(
    index: Dict[int, List[int]]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Map original identity labels (e.g. 0, 3, 700, …) to
    contiguous integers starting from 0 (required by the loss function).

    Returns
    -------
    remapped_index : {new_label: [record_ids]}
    label_map      : {original_label: new_label}
    """
    label_map: Dict[int, int] = {
        orig: new for new, orig in enumerate(sorted(index.keys()))
    }
    remapped_index: Dict[int, List[int]] = {
        label_map[orig]: rids for orig, rids in index.items()
    }
    return remapped_index, label_map


# ===================================================================
# Step 7 – tf.data.Dataset builder
# ===================================================================

def make_recordio_tf_dataset(
    rec_path: str,
    offset_map: Dict[int, int],
    identity_index: Dict[int, List[int]],   # already remapped to 0-based labels
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that lazily reads images from train.rec.

    How it works (plain English)
    ----------------------------
    1. We collect a flat list of (record_id, label) pairs from identity_index.
    2. tf.data.Dataset.from_generator() creates a pipeline that calls our
       Python generator function one sample at a time.
    3. The generator opens train.rec ONCE and keeps it open, seeking to each
       record's offset on demand.  No image is held in memory after it is
       yielded.
    4. TensorFlow batches the results automatically.

    Parameters
    ----------
    rec_path       : path to train.rec
    offset_map     : {record_id: byte_offset}
    identity_index : {new_label (0-based): [record_ids]}
    num_classes    : total number of classes in this split
    image_size     : (height, width) for output images
    batch_size     : samples per batch
    augment        : whether to apply random flips / brightness jitter
    shuffle        : whether to shuffle sample order each epoch
    seed           : reproducibility seed

    Returns
    -------
    dataset : tf.data.Dataset  yielding (image, one_hot_label) batches
              image shape : (batch, height, width, 3), dtype float32
              label shape : (batch, num_classes),       dtype float32
    """
    # Build flat list of (record_id, label) pairs
    pairs: List[Tuple[int, int]] = []
    for label, rids in identity_index.items():
        for rid in rids:
            if rid in offset_map:   # skip record_ids not in the index file
                pairs.append((rid, label))

    print(f"[RecordIO] Dataset: {len(pairs):,} samples, {num_classes} classes")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(pairs)

    # Store pairs as numpy arrays for the generator
    record_ids_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    labels_arr     = np.array([p[1] for p in pairs], dtype=np.int64)

    def _generator():
        """
        Generator that opens train.rec once and yields (image, label) pairs.
        Runs inside tf.data.Dataset.from_generator as a Python callable.
        """
        with open(rec_path, "rb") as rec_file:
            for rid, lbl in zip(record_ids_arr, labels_arr):
                try:
                    image, _ = read_record(
                        rec_file, offset_map, int(rid), target_size=image_size
                    )
                    # One-hot encode label
                    one_hot = np.zeros(num_classes, dtype=np.float32)
                    one_hot[int(lbl)] = 1.0
                    yield image.astype(np.float32), one_hot
                except Exception:
                    # Skip corrupted records gracefully
                    continue

    output_signature = (
        tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,),                     dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature=output_signature,
    )

    if augment:
        def _augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.15)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def create_recordio_train_val_test_splits(
    identity_index: Dict[int, List[int]],
    val_image_fraction: float = 0.15,
    test_identity_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Create a rigorous 3-way partition for RecordIO dataset:
      1. train_index : training images of training identities
      2. val_index   : held-out images of the SAME training identities (for early stopping & val_loss)
      3. test_index  : completely UNSEEN identities (identity-disjoint open set for verification)
    """
    rng = np.random.default_rng(seed)
    all_labels = np.array(sorted(identity_index.keys()))
    rng.shuffle(all_labels)

    n = len(all_labels)
    n_test = max(1, int(math.ceil(n * test_identity_fraction))) if (test_identity_fraction > 0 and n >= 4) else 0

    train_id_pool = all_labels[n_test:]
    test_id_pool = all_labels[:n_test]

    train_index = {}
    val_index = {}
    for lid in train_id_pool:
        rids = list(identity_index[int(lid)])
        rng.shuffle(rids)
        n_val = max(1, int(round(len(rids) * val_image_fraction))) if len(rids) >= 2 else 0
        if n_val > 0 and len(rids) - n_val >= 1:
            val_index[int(lid)] = rids[:n_val]
            train_index[int(lid)] = rids[n_val:]
        else:
            train_index[int(lid)] = rids
            val_index[int(lid)] = rids[:1]

    test_index = {int(lid): list(identity_index[int(lid)]) for lid in test_id_pool}

    print(
        f"[RecordIO] 3-Way Partition -> "
        f"train={len(train_index)} classes ({sum(len(v) for v in train_index.values())} imgs), "
        f"val={len(val_index)} classes ({sum(len(v) for v in val_index.values())} imgs), "
        f"test={len(test_index)} unseen classes ({sum(len(v) for v in test_index.values())} imgs)"
    )
    return train_index, val_index, test_index


# ===================================================================
# High-level convenience function
# ===================================================================

def load_recordio_splits(
    rec_path: str,
    idx_path: str,
    lst_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    max_identities: int = 100,
    min_images: int = 10,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, List[int]]:
    """
    One-stop function: parse LST + IDX, select identities, split, build datasets.

    Returns
    -------
    train_ds   : tf.data.Dataset  (shuffled, augmented)
    val_ds     : tf.data.Dataset  (not augmented, matching num_classes for val_loss)
    test_ds    : tf.data.Dataset  (not augmented, open-set unseen identities)
    num_classes: int — number of training classes
    train_labels: list of original identity labels used for training
    """
    # Parse files
    identity_index = build_identity_index(lst_path)
    offset_map     = build_offset_map(idx_path)

    # Select subset of identities
    selected = select_top_n_identities(
        identity_index, n=max_identities, min_images=min_images, seed=seed
    )

    # 3-way partition
    train_raw, val_raw, test_raw = create_recordio_train_val_test_splits(
        selected, val_image_fraction=val_fraction, test_identity_fraction=test_fraction, seed=seed
    )

    # Remap training classes to 0..N-1
    train_idx, label_map = remap_labels(train_raw)
    num_train_classes = len(train_idx)
    train_labels_list = sorted(train_raw.keys())

    # Validation uses identical class mapping to match loss_fn dimensions
    val_idx = {
        label_map[orig]: rids for orig, rids in val_raw.items() if orig in label_map
    }

    # Test identities are remapped for open-set evaluation
    test_idx, _ = remap_labels(test_raw)

    # Build tf.data.Datasets
    train_ds = make_recordio_tf_dataset(
        rec_path=rec_path, offset_map=offset_map,
        identity_index=train_idx, num_classes=num_train_classes,
        image_size=image_size, batch_size=batch_size,
        augment=True, shuffle=True, seed=seed,
    )
    val_ds = make_recordio_tf_dataset(
        rec_path=rec_path, offset_map=offset_map,
        identity_index=val_idx, num_classes=num_train_classes,
        image_size=image_size, batch_size=batch_size,
        augment=False, shuffle=False, seed=seed,
    )
    test_ds = make_recordio_tf_dataset(
        rec_path=rec_path, offset_map=offset_map,
        identity_index=test_idx, num_classes=len(test_idx),
        image_size=image_size, batch_size=batch_size,
        augment=False, shuffle=False, seed=seed,
    )

    return train_ds, val_ds, test_ds, num_train_classes, train_labels_list
