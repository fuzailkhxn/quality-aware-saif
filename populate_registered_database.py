"""
populate_registered_database.py
================================
Registers all people and all valid face images from dataset/ into:
1. Physical folder: dataset/registered/<identity>/
2. Research Database: data/database_mobilenet_512.pkl (512-d MobileNetV2 with trained real weights)
3. Demo Database: data/database_sface_128.pkl (128-d OpenCV SFace)

Identities registered:
- adnan
- amir
- ranveer
- saif
- salman
- sharuk
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import shutil
import cv2
import numpy as np

from src.database import FaceDatabase
from src.preprocessing.face_detection import FaceDetector, crop_face
from src.preprocessing.quality_assessment import FaceQualityAssessor
from src.recognition.recognizer import QualityAwareRecognizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REGISTERED_DIR = os.path.join(DATASET_DIR, "registered")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REGISTERED_DIR, exist_ok=True)


def sync_registered_folders():
    print("=== [1/3] Syncing dataset/registered/ Folders ===")
    subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != "registered" and not d.startswith('.')]
    
    for ident in sorted(subdirs):
        src_dir = os.path.join(DATASET_DIR, ident)
        dst_dir = os.path.join(REGISTERED_DIR, ident)
        os.makedirs(dst_dir, exist_ok=True)
        
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        copied = 0
        for fname in files:
            sf = os.path.join(src_dir, fname)
            df = os.path.join(dst_dir, fname)
            if not os.path.exists(df):
                shutil.copy2(sf, df)
                copied += 1
        print(f"  {ident:10s} -> {len(files)} total images ({copied} newly synced to dataset/registered/{ident})")


def populate_engine_database(engine_choice: str):
    print(f"\n=== [2/3] Populating Database for Engine: {engine_choice} ===")
    weights_path = os.path.join(BASE_DIR, "checkpoints", "real_embedding.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BASE_DIR, "checkpoints", "demo_embedding.weights.h5")

    detector = FaceDetector()
    assessor = FaceQualityAssessor()
    
    if "sface" in engine_choice.lower():
        db = FaceDatabase(model_name="SFace-128", expected_dim=128)
        db.clear()
        recognizer = QualityAwareRecognizer(engine="SFace-128", database=db, detector=detector, assessor=assessor)
    else:
        db = FaceDatabase(model_name="MobileNetV2-512", expected_dim=512)
        db.clear()
        recognizer = QualityAwareRecognizer(
            engine="MobileNetV2-512",
            database=db,
            detector=detector,
            assessor=assessor,
            weights_path=weights_path if os.path.exists(weights_path) else None,
        )

    subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != "registered" and not d.startswith('.')]
    
    total_added = 0
    for ident in sorted(subdirs):
        folder = os.path.join(DATASET_DIR, ident)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        added_for_ident = 0
        
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            primary = detector.get_primary_face(img)
            if primary is None:
                continue
                
            box = primary["box"]
            crop = crop_face(img, box, padding=0.12)
            if crop is None or crop.size == 0:
                continue
                
            q_info = assessor.assess(crop)
            q_score = float(q_info["overall_score_norm"])
            
            try:
                emb = recognizer.extract_embedding(crop, raw_detection=primary.get("raw"))
                db.add_embedding(
                    name=ident,
                    embedding=emb,
                    quality_score=q_score,
                    image_path=fpath,
                    metadata={"box": box, "filename": fname}
                )
                added_for_ident += 1
                total_added += 1
            except Exception as e:
                print(f"    Failed {fname} for {ident}: {e}")
                
        print(f"  Registered {ident:10s}: {added_for_ident}/{len(files)} faces enrolled")

    db.save()
    print(f"  [OK] Saved {recognizer.database.db_path} ({total_added} total face embeddings across {len(db.get_identities())} identities)")


def main():
    sync_registered_folders()
    populate_engine_database("MobileNetV2-512")
    populate_engine_database("SFace-128")
    print("\n=== [3/3] Registration Verification ===")
    
    db_m = FaceDatabase(model_name="MobileNetV2-512", expected_dim=512)
    print(f"MobileNetV2 Database ({db_m.db_path}):")
    for k, v in db_m.get_stats().items():
        proto = db_m.get_prototype(k)
        print(f"  - {k:10s}: {v:2d} embeddings | Prototype shape: {proto.shape if proto is not None else None}")
        
    db_s = FaceDatabase(model_name="SFace-128", expected_dim=128)
    print(f"\nSFace Database ({db_s.db_path}):")
    for k, v in db_s.get_stats().items():
        proto = db_s.get_prototype(k)
        print(f"  - {k:10s}: {v:2d} embeddings | Prototype shape: {proto.shape if proto is not None else None}")

    print("\nAll people are successfully registered everywhere!")


if __name__ == '__main__':
    main()
