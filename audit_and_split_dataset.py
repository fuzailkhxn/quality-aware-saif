import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import hashlib
import json
import random
import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

DATASET_DIR = r'dataset'
MANIFEST_PATH = os.path.join(DATASET_DIR, 'split_manifest.json')
YUNET_PATH = 'models/face_detection_yunet_2023mar.onnx'

detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH,
    config='',
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def scan_and_audit():
    print('=== DATASET COMPREHENSIVE INTEGRITY AUDIT ===')
    subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')]
    print(f'Discovered identity directories ({len(subdirs)}): {sorted(subdirs)}')
    
    hash_to_files = {}
    identity_records = {}
    
    for ident in sorted(subdirs):
        folder = os.path.join(DATASET_DIR, ident)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        records = []
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            fhash = compute_sha256(fpath)
            
            if fhash not in hash_to_files:
                hash_to_files[fhash] = []
            hash_to_files[fhash].append((ident, fname))
            
            img = cv2.imread(fpath)
            readable = img is not None
            face_detected = False
            face_box = None
            conf = 0.0
            
            if readable:
                h, w = img.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(img)
                if faces is not None and len(faces) > 0:
                    face_detected = True
                    conf = float(faces[0][-1])
                    face_box = [int(v) for v in faces[0][:4]]
            
            records.append({
                'filename': fname,
                'path': fpath.replace('\\', '/'),
                'sha256': fhash,
                'readable': readable,
                'face_detected': face_detected,
                'confidence': conf,
                'face_box': face_box
            })
            
        identity_records[ident] = records
        
    # Check duplicates (intra and cross)
    print('\n--- DUPLICATE ANALYSIS ---')
    dup_count = 0
    for h, flist in hash_to_files.items():
        if len(flist) > 1:
            dup_count += 1
            print(f'Duplicate found ({len(flist)} occurrences): {flist}')
    if dup_count == 0:
        print('ZERO duplicate images found (both intra-folder and cross-folder hashes are 100% unique).')
        
    print('\n--- PER-IDENTITY AUDIT SUMMARY ---')
    for ident, recs in identity_records.items():
        total = len(recs)
        readable_count = sum(1 for r in recs if r['readable'])
        detected_count = sum(1 for r in recs if r['face_detected'])
        unusable_count = total - detected_count
        print(f"Identity: {ident:10s} | Total: {total:2d} | Readable: {readable_count:2d} | Faces Detected: {detected_count:2d} | Unusable: {unusable_count:2d}")
        
    # Prepare zero-leakage splits for 4 training identities
    TRAIN_CLASSES = ['amir', 'saif', 'salman', 'sharuk']
    splits = {'train': [], 'val': [], 'test': [], 'unknown_probe': []}
    
    # Adnan is unknown probe
    for r in identity_records.get('adnan', []):
        if r['face_detected']:
            splits['unknown_probe'].append({**r, 'identity': 'adnan', 'label': -1})
            
    # For training classes
    for ident_idx, ident in enumerate(TRAIN_CLASSES):
        valid_recs = [r for r in identity_records[ident] if r['face_detected']]
        # Shuffle deterministically
        random.shuffle(valid_recs)
        
        n = len(valid_recs)
        if ident == 'saif':
            # 35 train, 7 val, 7 test
            n_train = 35
            n_val = 7
            n_test = 7
        elif ident in ['amir', 'salman']:
            # 5 train, 1 val, 1 test
            n_train = 5
            n_val = 1
            n_test = 1
        elif ident == 'sharuk':
            # 4 train, 1 val, 1 test
            n_train = 4
            n_val = 1
            n_test = 1
            
        train_items = valid_recs[:n_train]
        val_items = valid_recs[n_train:n_train + n_val]
        test_items = valid_recs[n_train + n_val:n_train + n_val + n_test]
        
        for item in train_items:
            splits['train'].append({**item, 'identity': ident, 'label': ident_idx})
        for item in val_items:
            splits['val'].append({**item, 'identity': ident, 'label': ident_idx})
        for item in test_items:
            splits['test'].append({**item, 'identity': ident, 'label': ident_idx})
            
    print('\n--- SPLIT VERIFICATION ---')
    print(f"Train set: {len(splits['train'])} images")
    print(f"Val set:   {len(splits['val'])} images")
    print(f"Test set:  {len(splits['test'])} images")
    print(f"Unknown:   {len(splits['unknown_probe'])} images")
    
    # Overlap / Leakage checks
    train_paths = {x['path'] for x in splits['train']}
    val_paths = {x['path'] for x in splits['val']}
    test_paths = {x['path'] for x in splits['test']}
    
    leakage_train_val = train_paths.intersection(val_paths)
    leakage_train_test = train_paths.intersection(test_paths)
    leakage_val_test = val_paths.intersection(test_paths)
    
    print(f"Leakage Train-Val:  {len(leakage_train_val)}")
    print(f"Leakage Train-Test: {len(leakage_train_test)}")
    print(f"Leakage Val-Test:   {len(leakage_val_test)}")
    assert len(leakage_train_val) == 0, 'Data leakage between Train and Val!'
    assert len(leakage_train_test) == 0, 'Data leakage between Train and Test!'
    assert len(leakage_val_test) == 0, 'Data leakage between Val and Test!'
    print('ZERO DATA LEAKAGE CONFIRMED: Train, Val, and Test sets are mutually exclusive.')
    
    # Per-class counts in each split
    print('\n--- PER-CLASS SPLIT DISTRIBUTION ---')
    for ident_idx, ident in enumerate(TRAIN_CLASSES):
        n_tr = sum(1 for x in splits['train'] if x['identity'] == ident)
        n_va = sum(1 for x in splits['val'] if x['identity'] == ident)
        n_te = sum(1 for x in splits['test'] if x['identity'] == ident)
        print(f"Class {ident_idx} ({ident:8s}): Train={n_tr:2d}, Val={n_va:2d}, Test={n_te:2d} (Total={n_tr+n_va+n_te})")
        
    manifest = {
        'classes': TRAIN_CLASSES,
        'class_to_idx': {c: i for i, c in enumerate(TRAIN_CLASSES)},
        'splits': splits,
        'summary': {
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
            'unknown_count': len(splits['unknown_probe']),
            'zero_leakage_verified': True
        }
    }
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest successfully saved to {MANIFEST_PATH}")

if __name__ == '__main__':
    scan_and_audit()
