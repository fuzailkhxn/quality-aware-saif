import os
import cv2
import hashlib

DATASET_DIR = "dataset"
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

print("=" * 60)
print("AUDITING ACTIVE 5 IDENTITIES IN DATASET")
print("=" * 60)

total_images = 0
total_usable = 0
unusable_files = []
hash_map = {}

for c in CLASSES:
    cdir = os.path.join(DATASET_DIR, c)
    if not os.path.isdir(cdir):
        print(f"ERROR: Directory not found: {cdir}")
        continue
    files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    c_usable = 0
    for fname in sorted(files):
        fpath = os.path.join(cdir, fname)
        total_images += 1
        fhash = compute_sha256(fpath)
        if fhash not in hash_map:
            hash_map[fhash] = []
        hash_map[fhash].append((c, fname))
        
        img = cv2.imread(fpath)
        if img is None:
            unusable_files.append((c, fname, "Unreadable image"))
            continue
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            c_usable += 1
            total_usable += 1
        else:
            unusable_files.append((c, fname, "Face detection failed"))
    print(f"Identity: {c:8s} | Files: {len(files):2d} | Usable Faces: {c_usable:2d} | Failed: {len(files) - c_usable}")

print("-" * 60)
print(f"Total Images across 5 classes: {total_images}")
print(f"Total Usable Faces:            {total_usable}")

duplicates = {h: flist for h, flist in hash_map.items() if len(flist) > 1}
if duplicates:
    print(f"Duplicates detected: {len(duplicates)}")
    for h, flist in duplicates.items():
        print(f"  {flist}")
else:
    print("Zero duplicates detected (all images unique).")

if unusable_files:
    print(f"Unusable images ({len(unusable_files)}):")
    for c, fn, reason in unusable_files:
        print(f"  {c}/{fn}: {reason}")
