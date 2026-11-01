"""
src/preprocessing/dataset_loader.py
===================================
Robust dataset loader and validator for CASIA-WebFace and identity image folders.

Features:
1. Validates and skips corrupted/unreadable/zero-byte images.
2. Filters out identities that have fewer than `min_images` samples.
3. Supports limiting the number of identities (`max_identities`) for fast experiments.
4. Supports identity-disjoint train/validation splitting to guarantee zero data leakage
   for open-set face verification evaluation (Objective E).
5. Builds optimized, parallelized `tf.data.Dataset` pipelines.
6. Includes a standalone RecordIO unpacker utility for MXNet .rec/.idx packages.
"""

import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg


def is_valid_image(image_path: str) -> bool:
    """
    Check if an image file exists, is non-empty, and can be read by PIL.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    bool
        True if image is completely valid, False if corrupted or unreadable.
    """
    if not os.path.isfile(image_path):
        return False

    # Check file size (> 1KB)
    try:
        if os.path.getsize(image_path) < 1024:
            return False
    except OSError:
        return False

    # Verify image integrity via PIL
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Ensure we can load image data (detects truncated files)
        with Image.open(image_path) as img:
            img.load()
            if img.size[0] < 10 or img.size[1] < 10:
                return False
        return True
    except Exception:
        return False


def is_recordio_dataset(dataset_path: str) -> bool:
    """
    Check if a directory contains MXNet RecordIO files (train.rec and train.idx).
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        return False
    rec_path = os.path.join(dataset_path, "train.rec")
    idx_path = os.path.join(dataset_path, "train.idx")
    return os.path.isfile(rec_path) and os.path.isfile(idx_path)


def scan_and_validate_dataset(
    dataset_path: str,
    min_images_per_identity: int = 10,
    max_identities: Optional[int] = None,
    sort_by_sample_count: bool = True,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Scan a dataset folder, validate all images, and filter by sample count.

    Parameters
    ----------
    dataset_path : str
        Path to root dataset folder containing one subfolder per identity.
    min_images_per_identity : int
        Minimum number of valid images required to include an identity.
    max_identities : int, optional
        Maximum number of identities to retain (useful for subset training).
    sort_by_sample_count : bool
        If True, selects identities with the most photos when max_identities is set.
    seed : int
        Random seed for shuffling if not sorting by sample count.

    Returns
    -------
    dict
        Mapping of identity_name -> list of verified valid image file paths.
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    raw_subfolders = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not raw_subfolders:
        if is_recordio_dataset(dataset_path):
            raise ValueError(
                f"Directory '{dataset_path}' is an MXNet RecordIO dataset (contains train.rec and train.idx), "
                f"not a folder-based image dataset. Please use RecordIO loading mode."
            )
        raise ValueError(f"No identity subfolders found inside '{dataset_path}'.")

    print(f"\n[Dataset Scanner] Scanning {len(raw_subfolders)} subfolders in '{dataset_path}'...")

    dataset_dict: Dict[str, List[str]] = {}
    total_corrupted = 0

    for idx, folder_name in enumerate(raw_subfolders):
        folder_path = os.path.join(dataset_path, folder_name)
        image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        valid_images = []
        for img_path in image_files:
            if is_valid_image(img_path):
                valid_images.append(img_path)
            else:
                total_corrupted += 1

        if len(valid_images) >= min_images_per_identity:
            dataset_dict[folder_name] = valid_images

    print(f"[Dataset Scanner] Validated {len(dataset_dict)} identities meeting min_images >= {min_images_per_identity}.")
    if total_corrupted > 0:
        print(f"[Dataset Scanner] Skipped {total_corrupted} corrupted or unreadable images.")

    if len(dataset_dict) < 2:
        raise ValueError(
            f"Found only {len(dataset_dict)} valid identities with at least {min_images_per_identity} images.\n"
            f"At least 2 identities are required for training and verification."
        )

    # Apply max_identities limit if specified
    if max_identities is not None and len(dataset_dict) > max_identities:
        if sort_by_sample_count:
            # Pick identities with the largest number of valid images
            sorted_identities = sorted(
                dataset_dict.keys(),
                key=lambda k: len(dataset_dict[k]),
                reverse=True,
            )[:max_identities]
        else:
            rng = random.Random(seed)
            keys = list(dataset_dict.keys())
            rng.shuffle(keys)
            sorted_identities = keys[:max_identities]

        dataset_dict = {k: dataset_dict[k] for k in sorted_identities}
        print(f"[Dataset Scanner] Filtered to top {max_identities} identities.")

    total_images = sum(len(imgs) for imgs in dataset_dict.values())
    print(f"[Dataset Scanner] Ready: {len(dataset_dict)} identities, {total_images} total images.\n")

    return dataset_dict


def create_identity_disjoint_splits(
    dataset_dict: Dict[str, List[str]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split the dataset such that validation identities are completely unseen in training.
    This provides an open-set benchmark for Face Verification (Objective E).

    Parameters
    ----------
    dataset_dict : dict
        Mapping of identity_name -> list of image paths.
    val_fraction : float
        Fraction of identities to hold out for validation (e.g. 0.15 = 15%).
    seed : int
        Random seed for reproducible split.

    Returns
    -------
    (train_dict, val_dict)
    """
    identities = sorted(list(dataset_dict.keys()))
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_val = max(1, int(len(identities) * val_fraction))
    val_identities = set(identities[:n_val])
    train_identities = set(identities[n_val:])

    # Guarantee at least 2 training classes
    if len(train_identities) < 2:
        val_identities = set(identities[:1])
        train_identities = set(identities[1:])

    train_dict = {k: dataset_dict[k] for k in train_identities}
    val_dict = {k: dataset_dict[k] for k in val_identities}

    print(f"[Split] Identity-Disjoint Split Created:")
    print(f"  Training   : {len(train_dict)} identities, {sum(len(v) for v in train_dict.values())} images")
    print(f"  Validation : {len(val_dict)} identities, {sum(len(v) for v in val_dict.values())} images (Unseen IDs)")

    return train_dict, val_dict


def create_training_and_verification_splits(
    dataset_dict: Dict[str, List[str]],
    test_identity_fraction: float = 0.15,
    train_val_image_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Creates standard face recognition splits:
      1. train_dict: Training images for identity prototype learning.
      2. train_val_dict: Held-out images of the SAME training identities
         (used by trainer for val_loss / early stopping).
      3. verification_dict: Completely UNSEEN identities (used for open-set
         ROC curve, AUC, EER, and TAR@FAR evaluation).

    Parameters
    ----------
    dataset_dict : dict
        Mapping of identity_name -> list of image paths.
    test_identity_fraction : float
        Fraction of identities to reserve for open-set verification.
    train_val_image_fraction : float
        Fraction of images within training identities for early stopping.
    seed : int

    Returns
    -------
    (train_dict, train_val_dict, verification_dict)
    """
    rng = random.Random(seed)
    identities = sorted(list(dataset_dict.keys()))

    if len(identities) < 4:
        # For small test datasets with < 4 identities, use closed-set split
        train_dict, train_val_dict = create_closed_set_splits(
            dataset_dict, val_fraction=train_val_image_fraction, seed=seed
        )
        return train_dict, train_val_dict, train_val_dict

    # Split identities
    rng.shuffle(identities)
    n_test_ids = max(1, int(len(identities) * test_identity_fraction))
    test_ids = set(identities[:n_test_ids])
    train_ids = set(identities[n_test_ids:])

    verification_dict = {k: dataset_dict[k] for k in test_ids}

    # Split train identities into train images and val images
    train_dict = {}
    train_val_dict = {}
    for identity in train_ids:
        imgs = list(dataset_dict[identity])
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * train_val_image_fraction))
        train_val_dict[identity] = imgs[:n_val]
        train_dict[identity] = imgs[n_val:]

    print(f"[Dataset Split] Rigorous 3-way partition created:")
    print(f"  Training       : {len(train_dict)} identities, {sum(len(v) for v in train_dict.values())} images")
    print(f"  Train-Val Loss : {len(train_val_dict)} identities, {sum(len(v) for v in train_val_dict.values())} images")
    print(f"  Verification   : {len(verification_dict)} UNSEEN identities, {sum(len(v) for v in verification_dict.values())} images (Zero Leakage)")

    return train_dict, train_val_dict, verification_dict


def create_closed_set_splits(
    dataset_dict: Dict[str, List[str]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Standard closed-set split where every identity appears in both train and val.
    """
    rng = random.Random(seed)
    train_dict = {}
    val_dict = {}

    for identity, img_paths in dataset_dict.items():
        shuffled = list(img_paths)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_dict[identity] = shuffled[:n_val]
        train_dict[identity] = shuffled[n_val:]

    return train_dict, val_dict


def _load_and_preprocess_image(file_path: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    """TensorFlow native graph image loading and normalisation."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def build_tf_dataset_from_dict(
    dataset_dict: Dict[str, List[str]],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    is_training: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, List[str], int]:
    """
    Convert an identity->images dictionary into a high-performance tf.data.Dataset.

    Parameters
    ----------
    dataset_dict : dict
        Mapping of identity_name -> list of image paths.
    image_size : tuple (H, W)
    batch_size : int
    augment : bool
        Whether to apply data augmentation.
    is_training : bool
        Whether to shuffle and apply repeat/augmentation.
    seed : int

    Returns
    -------
    (tf_dataset, class_names, num_classes)
    """
    class_names = sorted(list(dataset_dict.keys()))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    all_paths = []
    all_labels = []

    for name, paths in dataset_dict.items():
        label_idx = class_to_idx[name]
        for p in paths:
            all_paths.append(p)
            all_labels.append(label_idx)

    # Shuffle paired list
    combined = list(zip(all_paths, all_labels))
    if is_training:
        rng = random.Random(seed)
        rng.shuffle(combined)

    paths_clean, labels_clean = zip(*combined) if combined else ([], [])
    paths_tensor = tf.constant(list(paths_clean), dtype=tf.string)
    labels_tensor = tf.one_hot(list(labels_clean), depth=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths_clean), 2048), seed=seed)

    # Map image loading
    h, w = image_size
    ds = ds.map(
        lambda p, y: (_load_and_preprocess_image(p, (h, w)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment and is_training:
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(cfg.AUGMENTATION["rotation_range"] / 360.0),
            tf.keras.layers.RandomZoom(cfg.AUGMENTATION["zoom_range"]),
            tf.keras.layers.RandomTranslation(
                cfg.AUGMENTATION["height_shift_range"],
                cfg.AUGMENTATION["width_shift_range"],
            ),
        ], name="augmentation")
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=is_training and len(paths_clean) >= batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names, num_classes


# ---------------------------------------------------------------------------
# Optional: Standalone MXNet RecordIO Unpacker
# ---------------------------------------------------------------------------

def unpack_recordio(rec_path: str, idx_path: str, output_dir: str) -> int:
    """
    Unpack an MXNet RecordIO dataset (.rec/.idx) into directory structure
    without requiring the MXNet package.

    Parameters
    ----------
    rec_path : str
        Path to .rec file.
    idx_path : str
        Path to .idx file.
    output_dir : str
        Target output directory (e.g. dataset/casia_webface).

    Returns
    -------
    int
        Total extracted images.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[RecordIO] Unpacking {rec_path} to {output_dir}...")

    # Reads standard MXNet indexed record
    import struct
    total_images = 0

    with open(idx_path, "r") as f_idx, open(rec_path, "rb") as f_rec:
        for line in f_idx:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            offset = int(parts[1])

            f_rec.seek(offset)
            magic = struct.unpack("I", f_rec.read(4))[0]
            if magic != 0xced7230a:
                continue

            length_flag = struct.unpack("I", f_rec.read(4))[0]
            cflag = length_flag >> 29
            length = length_flag & 0x1fffffff

            data = f_rec.read(length)
            # The header stores [flag, label, id, id2]
            header_size = 4 * 4
            if length <= header_size:
                continue

            header = struct.unpack("4I", data[:header_size])
            label = header[1]
            image_bytes = data[header_size:]

            id_dir = os.path.join(output_dir, f"{label:07d}")
            os.makedirs(id_dir, exist_ok=True)

            out_file = os.path.join(id_dir, f"{idx:07d}.jpg")
            with open(out_file, "wb") as f_out:
                f_out.write(image_bytes)
            total_images += 1

    print(f"[RecordIO] Extracted {total_images} images successfully.")
    return total_images


def load_split_from_manifest(
    manifest_path: str = "dataset/split_manifest.json"
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    Load train, val, test, and unknown splits from a JSON manifest.

    Parameters
    ----------
    manifest_path : str
        Path to split_manifest.json.

    Returns
    -------
    (train_dict, val_dict, test_dict, unknown_dict, class_names)
    """
    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    splits = manifest["splits"]

    def _group_by_identity(item_list):
        grouped: Dict[str, List[str]] = {}
        for item in item_list:
            ident = item["identity"]
            path = item["path"]
            if ident not in grouped:
                grouped[ident] = []
            grouped[ident].append(path)
        return grouped

    train_dict = _group_by_identity(splits["train"])
    val_dict = _group_by_identity(splits["val"])
    test_dict = _group_by_identity(splits["test"])
    unknown_dict = _group_by_identity(splits.get("unknown_probe", []))

    return train_dict, val_dict, test_dict, unknown_dict, classes

