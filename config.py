"""
config.py
=========
Central configuration for Quality-Aware Face Recognition.

All hyperparameters are defined here. Do not hardcode values in other modules.
Edit this file to change training behaviour without touching model code.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths  (relative to project root; trainer resolves them to absolute paths)
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset"
CASIA_DATASET_PATH = "dataset/casia_webface"
MODEL_SAVE_PATH = "models/face_model.h5"
EMBEDDING_MODEL_PATH = "models/embedding_model.keras"
EMBEDDING_WEIGHTS_PATH = "models/embedding_model.weights.h5"
PROTOTYPE_SAVE_PATH = "models/prototypes.npy"
RESULTS_DIR = "results"
REPORTS_DIR = "reports"
LOGS_DIR = "results/logs"

# ---------------------------------------------------------------------------
# Dataset Filtering & Scalability (CASIA-WebFace & Multi-Identity datasets)
# ---------------------------------------------------------------------------
MIN_IMAGES_PER_IDENTITY = 10      # Skip identities with fewer than this many photos
MAX_IDENTITIES = 100              # None = all identities; integer = subset of top identities
IDENTITY_DISJOINT_SPLIT = True    # True = zero-leakage open-set split (recommended)
VAL_IDENTITY_FRACTION = 0.15      # Fraction of identities reserved for validation / ROC eval

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # (height, width) — MobileNetV2 input
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 512              # Output dimension of the embedding head
BACKBONE = "MobileNetV2"         # Pretrained backbone name

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2           # Fraction of data used for validation
FINE_TUNE_AT_EPOCH = 15          # Epoch at which backbone unfreezing starts
FINE_TUNE_LAYERS = 30            # Number of top backbone layers to unfreeze

# ---------------------------------------------------------------------------
# Prototype Memory  (Objective A + B)
# ---------------------------------------------------------------------------
PROTOTYPE_MOMENTUM = 0.9         # α in update rule: P ← (1-α·w)·P + α·w·f
                                 # Higher → prototypes change slowly (stable)
PROTOTYPE_INIT_BATCHES = 5       # How many initial batches to collect before
                                 # first prototype initialisation

# ---------------------------------------------------------------------------
# Adaptive Margin Loss  (Objective C)
# ---------------------------------------------------------------------------
ARC_SCALE = 64.0                 # Scaling factor s in ArcFace
ARC_BASE_MARGIN = 0.35           # Base margin m (radians)
ADAPTIVE_MARGIN_LAMBDA = 0.5     # λ: controls how much sample difficulty
                                 # adjusts the margin
                                 # margin_i = m_base + λ·(1 − cos(f_i, P_c))
MARGIN_CLIP_MIN = 0.1            # Minimum allowed margin (numerical safety)
MARGIN_CLIP_MAX = 0.7            # Maximum allowed margin

# ---------------------------------------------------------------------------
# Noisy / Hard Sample Handling  (Objective D)
# ---------------------------------------------------------------------------
NOISE_WEIGHT_FLOOR = 0.1         # Minimum loss weight for any sample
                                 # (noisy samples still contribute a little)
QUALITY_WEIGHT_ALPHA = 0.6       # Weight of quality score in sample weight
CONFIDENCE_WEIGHT_ALPHA = 0.4    # Weight of prediction confidence in sample weight

# ---------------------------------------------------------------------------
# Quality Assessment  (supports Objective D)
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 100.0           # Laplacian variance below this → blurry image
BRIGHTNESS_LOW = 40              # Pixel mean below this → too dark
BRIGHTNESS_HIGH = 220            # Pixel mean above this → overexposed

# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------
AUGMENTATION = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# ---------------------------------------------------------------------------
# Evaluation  (Objective E)
# ---------------------------------------------------------------------------
FAR_THRESHOLDS = [0.001, 0.01, 0.1]   # TAR will be reported at these FAR values
                                       # 0.001 = FAR 0.1%,  0.01 = FAR 1%
VERIFICATION_THRESHOLD = 0.5          # Default cosine similarity threshold

# ---------------------------------------------------------------------------
# Derived / convenience
# ---------------------------------------------------------------------------
def get_abs_path(relative_path: str) -> str:
    """Resolve a config path relative to this file's directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


# ---------------------------------------------------------------------------
# RecordIO Dataset (CASIA-WebFace MXNet format)  — added for RecordIO support
# ---------------------------------------------------------------------------
# Absolute path to the directory containing train.rec, train.idx, train.lst
# and the .bin verification benchmark files.
# Edit this to point to wherever you extracted faces_webface_112x112.
RECORDIO_DATASET_PATH = r"C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112"

# Set to True to use RecordIO mode (--recordio flag in train.py sets this too)
USE_RECORDIO = False

# Native stored image size in the .rec file (112×112 for CASIA-WebFace)
RECORDIO_STORED_SIZE = (112, 112)

# Fraction of identities held out for the TEST split in RecordIO mode
# (VAL_IDENTITY_FRACTION above is reused for the validation split)
RECORDIO_TEST_FRACTION = 0.15
