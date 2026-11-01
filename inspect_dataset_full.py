import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import hashlib
import cv2
import numpy as np

yunet_path = 'models/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config='',
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

def inspect_folder(base_dir):
    print('=== Inspecting:', base_dir)
    if not os.path.exists(base_dir):
        print('Directory does not exist!')
        return
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != 'registered']
    print(f'Found {len(subdirs)} identity directories: {subdirs}')
    
    seen_hashes = {}
    for d in sorted(subdirs):
        folder_path = os.path.join(base_dir, d)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(files)
        readable = 0
        detected = 0
        duplicates = 0
        unusable = 0
        
        for f in files:
            fp = os.path.join(folder_path, f)
            try:
                with open(fp, 'rb') as fp_obj:
                    h = hashlib.sha256(fp_obj.read()).hexdigest()
                if h in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes[h] = fp
            except Exception:
                unusable += 1
                continue
                
            img = cv2.imread(fp)
            if img is None:
                unusable += 1
                continue
            readable += 1
            
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                detected += 1
            else:
                unusable += 1
                
        print(f'Identity: {d:10s} | Total: {total:2d} | Readable: {readable:2d} | Detected: {detected:2d} | Duplicates: {duplicates:2d} | Unusable: {unusable:2d}')

if __name__ == '__main__':
    inspect_folder(r'c:\Users\fuzail\Downloads\quality-aware-face-recognition-share\dataset')
    print()
    inspect_folder(r'c:\Users\fuzail\Downloads\Quality-Aware-Face-Recognition-System-main-old\Quality-Aware-Face-Recognition-System-main\dataset')
