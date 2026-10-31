# Quality-Aware Face Recognition

**AI-Based Face Recognition using Adaptive Empirical Prototype Learning with Confidence-Driven Dynamic Margins**

A research-grade face recognition system that improves accuracy over standard classifiers by:
- Maintaining per-identity prototype embeddings updated during training
- Using adaptive margins based on sample difficulty
- Down-weighting noisy/blurry training images automatically
- Evaluating robustness via ROC, AUC, EER and TAR@FAR metrics

---

## 👥 Team Members

- Adnan Ansari
- Saif
- Fuzail

---

## 🏗️ Project Structure

```
quality-aware-face-recognition-main/
│
├── config.py                      ← All hyperparameters (edit here)
├── train.py                       ← Main training script
├── requirements.txt               ← Python dependencies
│
├── src/
│   ├── face_model.py              ← Model definitions (baseline + embedding)
│   ├── image_processing.py        ← Image loading, preprocessing, face detection
│   │
│   ├── models/
│   │   ├── embedding_model.py     ← Re-export of embedding model
│   │   └── prototype_memory.py    ← Adaptive prototype storage & updates
│   │
│   ├── training/
│   │   ├── losses.py              ← Adaptive margin loss (ArcFace-style)
│   │   └── trainer.py             ← Custom training loop
│   │
│   ├── preprocessing/
│   │   ├── quality_assessment.py  ← Image sharpness + brightness scoring
│   │   ├── dataset_loader.py      ← Folder-based dataset scanning & validation
│   │   ├── recordio_loader.py     ← MXNet RecordIO reader (train.rec / idx / lst)
│   │   └── bin_loader.py          ← Benchmark verification loader (.bin)
│   │
│   ├── evaluation/
│   │   └── evaluate.py            ← ROC, AUC, EER, TAR@FAR evaluation
│   │
│   └── utils/
│       ├── metrics.py             ← Cosine similarity, EER helpers
│       └── seed_utils.py          ← Reproducibility (random seed control)
│
├── dataset/                       ← Face images (one subfolder per person)
│   └── adnan/
│       └── person.jpg
│
├── models/                        ← Saved model weights
│   ├── face_model.h5              ← Baseline model
│   ├── embedding_model.h5         ← Quality-aware embedding model (after training)
│   └── prototypes.npy             ← Prototype memory (after training)
│
├── results/
│   └── logs/
│       ├── training_log.csv       ← Epoch-by-epoch metrics
│       └── baseline_log.csv
│
├── reports/
│   ├── roc_curve.png              ← ROC curve plot (after evaluation)
│   └── verification_report.json  ← Numeric evaluation results
│
└── tests/
    ├── test_prototypes.py
    ├── test_losses.py
    ├── test_evaluation.py
    ├── test_dataset_loader.py
    └── test_recordio_loader.py
```

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.9 – 3.11
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for GPU users:** Replace `tensorflow` with `tensorflow-gpu` in `requirements.txt`
> and install the matching CUDA + cuDNN for your GPU.

---

## 📁 Dataset Preparation

The system expects this folder structure:

```
dataset/
  person_name_1/
    img1.jpg
    img2.jpg
    ...  (aim for 20+ images per person)
  person_name_2/
    img1.jpg
    ...
  ... (at least 2 people required)
```

**Minimum requirements:**
- At least **2 different people** (subfolders)
- At least **10 images per person** (20+ recommended)
- Images should show the face clearly at various angles and lighting

**Supported formats:** `.jpg`, `.jpeg`, `.png`

### Example: Adding yourself

```
dataset/
  your_name/
    photo1.jpg
    photo2.jpg
    ...
```

---

## 🚀 Training

### Quality-Aware Training (Recommended)

```bash
python train.py
```

This runs the full quality-aware pipeline on the folder-based dataset:
1. Computes image quality scores per batch
2. Trains with adaptive margin loss
3. Updates prototypes after each batch
4. Evaluates on validation set at the end

### CASIA-WebFace RecordIO Training (Direct .rec / .idx)

To train directly on the **CASIA-WebFace MXNet RecordIO** dataset without extracting images to folders:

```bash
# Fast test experiment with 100 identities (recommended first run):
python train.py --recordio --max-identities 100 --epochs 5 --no-eval

# Full training with all 6 verification benchmarks (LFW, AgeDB-30, CALFW, CFP-FF, CFP-FP, CPLFW):
python train.py --recordio --max-identities 100 --epochs 30
```

*(See [README_RECORDIO.md](README_RECORDIO.md) for full RecordIO documentation).*

### Baseline Training (for comparison)

```bash
python train.py --baseline
```

### Options

```
python train.py --help

  --baseline          Use original classifier instead of quality-aware model
  --dataset PATH      Override dataset directory path
  --epochs N          Override number of training epochs
  --seed N            Set random seed (default: 42)
  --no-eval           Skip evaluation after training
```

---

## 🔍 Face Recognition & Verification Application

Use `recognize.py` to run verification or identification on new images:

### 1. Face Verification (1-to-1 Match)
Compare two photos to determine whether they belong to the same person:
```bash
python recognize.py --verify path/to/image1.jpg path/to/image2.jpg
```

### 2. Face Identification (1-to-N Search)
Identify an unknown face against learned prototypes:
```bash
python recognize.py --identify path/to/image.jpg
```

### 3. Baseline Model Identification
Test using the original baseline model:
```bash
python recognize.py --baseline --identify path/to/image.jpg
```

### Configuration

All hyperparameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_DIM` | 512 | Embedding vector size |
| `EPOCHS` | 30 | Max training epochs |
| `BATCH_SIZE` | 32 | Images per batch |
| `PROTOTYPE_MOMENTUM` | 0.9 | Prototype stability (0-1) |
| `ARC_BASE_MARGIN` | 0.35 | Base angular margin (radians) |
| `ADAPTIVE_MARGIN_LAMBDA` | 0.5 | How much difficulty adjusts margin |
| `BLUR_THRESHOLD` | 100.0 | Laplacian variance for "sharp" image |

---

## 📊 Evaluation

Evaluation runs automatically after training. Metrics reported:

| Metric | Meaning |
|---|---|
| **AUC** | Area Under ROC Curve. 1.0 = perfect, 0.5 = random |
| **EER** | Equal Error Rate. Lower is better |
| **TAR@FAR=0.1%** | How many genuine pairs accepted when only 0.1% impostor pairs accepted |
| **TAR@FAR=1%** | Same at FAR=1% |
| **Prototype Accuracy** | Nearest-prototype classification accuracy |

Outputs saved to `reports/`:
- `roc_curve.png` — ROC curve plot
- `verification_report.json` — All numeric results

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_prototypes.py -v
python -m pytest tests/test_losses.py -v
python -m pytest tests/test_evaluation.py -v
```

---

## 🔬 Research Objectives & Implementation

| Objective | Implementation |
|---|---|
| **A. Adaptive Empirical Prototypes** | `src/models/prototype_memory.py` — `PrototypeMemory` class; L2-normalised class-mean prototypes initialised from training embeddings |
| **B. Dynamic Prototype Updates** | Quality-weighted EMA: `P_c ← norm((1-α·w)·P_c + α·w·f_i)` in `PrototypeMemory.update()` |
| **C. Adaptive Margin Learning** | `src/training/losses.py` — `margin_i = clip(m_base + λ·(1-cos(f_i,P_c)), m_min, m_max)` |
| **D. Noisy Sample Handling** | Sample weight = `quality_score × confidence`; floor at `NOISE_WEIGHT_FLOOR` so noisy samples still contribute |
| **E. Robust Face Verification** | `src/evaluation/evaluate.py` — genuine/impostor pair generation, ROC curve, AUC, EER, TAR@FAR |

---

## 🛠️ Technologies

- Python 3.9+
- TensorFlow / Keras
- OpenCV
- NumPy
- scikit-learn (ROC, AUC)
- Matplotlib (plots)
- pytest (unit tests)

---

## 📜 License

This project is developed for academic and educational purposes.
