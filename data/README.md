# Data Directory

This directory stores persisted face embedding databases and evaluation artifacts.

## Files

- `database_mobilenet_512.pkl`: Serialized face embeddings (512-dimensional L2-normalized vectors) and metadata for the MobileNetV2 Research Engine.
- `database_sface_128.pkl`: Serialized face embeddings (128-dimensional vectors) for the OpenCV SFace Lightweight Demo Engine.

Embeddings from different models are strictly isolated into distinct database stores to prevent feature dimension and metric space incompatibilities.
