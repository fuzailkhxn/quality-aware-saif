import os
import hashlib
import json
import random
import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

DATASET_DIR = "dataset"
MANIFEST_PATH = os.path.join(DATASET_DIR, "split_manifest.json")
CLASSES = ["amir", "ranbir", "saif", "salman", "sharuk"]
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate_manifest():
    print("=" * 65)
    print("REGENERATING DATASET SPLIT MANIFEST FOR 5 IDENTITIES")
    print("=" * 65)

    hash_to_files = {}
    records_per_class = {}

    for c in CLASSES:
        cdir = os.path.join(DATASET_DIR, c)
        files = sorted([f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        valid_records = []
        for fname in files:
            fpath = os.path.join(cdir, fname)
            fhash = compute_sha256(fpath)
            if fhash not in hash_to_files:
                hash_to_files[fhash] = []
            hash_to_files[fhash].append((c, fname))

            img = cv2.imread(fpath)
            if img is None:
                continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                conf = float(faces[0][-1])
                box = [int(v) for v in faces[0][:4]]
                valid_records.append({
                    "filename": fname,
                    "path": fpath.replace("\\", "/"),
                    "sha256": fhash,
                    "confidence": conf,
                    "face_box": box
                })
        records_per_class[c] = valid_records
        print(f"Loaded {c:8s}: {len(valid_records)} usable images")

    # Duplicate check
    dups = [fl for fl in hash_to_files.values() if len(fl) > 1]
    if dups:
        print(f"WARNING: Duplicates found: {dups}")
    else:
        print("Zero duplicates detected across all 5 classes.")

    # Stratified Splits: ~70% Train, ~15% Val, ~15% Test
    # amir (63): 44 train, 9 val, 10 test
    # ranbir (59): 41 train, 9 val, 9 test
    # saif (49): 35 train, 7 val, 7 test
    # salman (61): 43 train, 9 val, 9 test
    # sharuk (60): 42 train, 9 val, 9 test
    split_targets = {
        "amir": (44, 9, 10),
        "ranbir": (41, 9, 9),
        "saif": (35, 7, 7),
        "salman": (43, 9, 9),
        "sharuk": (42, 9, 9),
    }

    splits = {"train": [], "val": [], "test": [], "unknown_probe": []}

    for idx, c in enumerate(CLASSES):
        recs = records_per_class[c]
        # Deterministic shuffle
        rng = random.Random(42 + idx)
        shuffled = list(recs)
        rng.shuffle(shuffled)

        n_train, n_val, n_test = split_targets[c]
        assert len(shuffled) == n_train + n_val + n_test, f"Count mismatch for {c}: {len(shuffled)} vs {n_train + n_val + n_test}"

        train_set = shuffled[:n_train]
        val_set = shuffled[n_train:n_train + n_val]
        test_set = shuffled[n_train + n_val:n_train + n_val + n_test]

        for item in train_set:
            splits["train"].append({**item, "identity": c, "label": idx})
        for item in val_set:
            splits["val"].append({**item, "identity": c, "label": idx})
        for item in test_set:
            splits["test"].append({**item, "identity": c, "label": idx})

    # Rigorous leakage check
    train_paths = {x["path"] for x in splits["train"]}
    val_paths = {x["path"] for x in splits["val"]}
    test_paths = {x["path"] for x in splits["test"]}

    leak_tv = train_paths.intersection(val_paths)
    leak_tt = train_paths.intersection(test_paths)
    leak_vt = val_paths.intersection(test_paths)

    assert len(leak_tv) == 0, f"Leakage Train-Val: {leak_tv}"
    assert len(leak_tt) == 0, f"Leakage Train-Test: {leak_tt}"
    assert len(leak_vt) == 0, f"Leakage Val-Test: {leak_vt}"

    train_hashes = {x["sha256"] for x in splits["train"]}
    val_hashes = {x["sha256"] for x in splits["val"]}
    test_hashes = {x["sha256"] for x in splits["test"]}
    assert len(train_hashes.intersection(val_hashes)) == 0, "Hash overlap Train-Val!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Hash overlap Train-Test!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Hash overlap Val-Test!"

    print("\n--- SPLIT STATISTICS ---")
    print(f"Train set: {len(splits['train']):3d} samples ({len(splits['train'])/292*100:.1f}%)")
    print(f"Val set:   {len(splits['val']):3d} samples ({len(splits['val'])/292*100:.1f}%)")
    print(f"Test set:  {len(splits['test']):3d} samples ({len(splits['test'])/292*100:.1f}%)")
    print(f"Total:     {len(splits['train']) + len(splits['val']) + len(splits['test']):3d} samples")

    print("\n--- PER-CLASS SPLITS ---")
    for idx, c in enumerate(CLASSES):
        tr = sum(1 for x in splits["train"] if x["identity"] == c)
        va = sum(1 for x in splits["val"] if x["identity"] == c)
        te = sum(1 for x in splits["test"] if x["identity"] == c)
        print(f"  {c:8s} -> Train: {tr:2d}, Val: {va:2d}, Test: {te:2d} (Total: {tr+va+te:2d})")

    manifest = {
        "classes": CLASSES,
        "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
        "splits": splits,
        "summary": {
            "total_images": 292,
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "zero_leakage_verified": True
        }
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to: {MANIFEST_PATH}")

if __name__ == "__main__":
    generate_manifest()
