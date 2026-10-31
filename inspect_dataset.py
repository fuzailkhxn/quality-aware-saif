import os
import sys
import hashlib
import cv2
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.preprocessing.face_detection import FaceDetector, crop_face
from src.preprocessing.quality_assessment import FaceQualityAssessor

dataset_path = os.path.abspath('dataset')
detector = FaceDetector()
assessor = FaceQualityAssessor()

print(f"=== INSPECTING DATASET AT: {dataset_path} ===")

identities = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print(f"Found {len(identities)} identity subfolders: {identities}")

total_images = 0
results_by_identity = {}

for ident in sorted(identities):
    ident_dir = os.path.join(dataset_path, ident)
    files = [f for f in os.listdir(ident_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"\n--- Identity: {ident} (Total files: {len(files)}) ---")
    
    hashes = {}
    readable_count = 0
    face_detected_count = 0
    corrupted_count = 0
    duplicate_count = 0
    quality_scores = []
    no_face_files = []
    
    for f in sorted(files):
        p = os.path.join(ident_dir, f)
        total_images += 1
        
        # Check readability & hash
        try:
            with open(p, 'rb') as fp:
                content = fp.read()
                file_hash = hashlib.sha256(content).hexdigest()
            if file_hash in hashes:
                duplicate_count += 1
                print(f"  [Duplicate] {f} matches {hashes[file_hash]}")
            else:
                hashes[file_hash] = f
                
            img = cv2.imread(p)
            if img is None or img.size == 0:
                corrupted_count += 1
                print(f"  [Corrupted] Could not decode image: {f}")
                continue
            readable_count += 1
            
            # Face detection
            faces = detector.detect_faces(img)
            # YuNet / fallback returns dicts with 'box', 'confidence'
            if faces and faces[0].get('confidence', 0) >= 0.5:
                face_detected_count += 1
                box = faces[0]['box']
                crop = crop_face(img, box)
                q = assessor.assess(crop)
                quality_scores.append(q['overall_score'])
            else:
                conf = faces[0]['confidence'] if faces else 0.0
                no_face_files.append((f, img.shape, conf))
                print(f"  [No Face/Low Conf] {f} (shape: {img.shape}, conf: {conf:.2f})")
        except Exception as e:
            corrupted_count += 1
            print(f"  [Exception] {f}: {e}")
            
    mean_q = float(np.mean(quality_scores)) if quality_scores else 0.0
    results_by_identity[ident] = {
        'total_files': len(files),
        'readable': readable_count,
        'duplicates': duplicate_count,
        'corrupted': corrupted_count,
        'faces_detected': face_detected_count,
        'mean_quality': round(mean_q, 1),
        'min_quality': round(float(np.min(quality_scores)), 1) if quality_scores else 0.0,
        'max_quality': round(float(np.max(quality_scores)), 1) if quality_scores else 0.0,
        'no_face_files': no_face_files,
    }

print("\n================ SUMMARY REPORT ================")
print(f"Dataset root path: {dataset_path}")
print(f"Total identity folders: {len(identities)}")
print(f"Total images scanned: {total_images}")
for ident, r in results_by_identity.items():
    print(f"\nIdentity '{ident}':")
    print(f"  Files on disk:     {r['total_files']}")
    print(f"  Readable images:   {r['readable']}")
    print(f"  Duplicates:        {r['duplicates']}")
    print(f"  Corrupted:         {r['corrupted']}")
    print(f"  Faces detected:    {r['faces_detected']} / {r['readable']}")
    print(f"  Quality Range:     {r['min_quality']} - {r['max_quality']} (Mean: {r['mean_quality']}/100)")
    if r['no_face_files']:
        print(f"  Uncertain faces ({len(r['no_face_files'])}): {r['no_face_files']}")
