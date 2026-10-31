# Models Directory

This directory stores deep learning backbone models, weights, and empirical prototype matrices.

## Model Files

1. `face_detection_yunet_2023mar.onnx` (232 KB)
   - Lightweight high-performance face detector from OpenCV Zoo.
   - Extracts bounding box `(x, y, w, h)`, 5-point facial landmarks, and detection confidence scores.

2. `face_recognition_sface_2021dec.onnx` (38 MB)
   - Pretrained SFace feature extractor from OpenCV Zoo.
   - Outputs 128-dimensional normalized facial feature representations for rapid CPU demonstration inference.

3. `face_model.h5` / `embedding_model.weights.h5`
   - Keras MobileNetV2 feature extractor backbone generating 512-dimensional normalized embeddings for research-grade verification.

4. `prototypes.npy` / `demo_prototypes.npy`
   - Learned empirical class prototypes $\mathbf{P}_c \in \mathbb{R}^{512}$ representing identity centers on the unit hypersphere.
