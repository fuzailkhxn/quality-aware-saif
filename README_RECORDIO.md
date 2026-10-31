# RecordIO Dataset Integration Guide

This document explains how to use your **CASIA-WebFace** dataset (in MXNet RecordIO
format) with the Quality-Aware Face Recognition training pipeline.

---

## What Files Are Required

Your dataset folder (`faces_webface_112x112`) must contain:

| File | Purpose |
|---|---|
| `train.rec` | All face images packed into one binary file |
| `train.idx` | Byte-offset index: tells us where each image is inside train.rec |
| `train.lst` | Identity labels: maps each image to a person ID (0–10571) |
| `lfw.bin` | LFW verification benchmark (optional but recommended) |
| `agedb_30.bin` | AgeDB-30 age-variation benchmark |
| `calfw.bin` | Cross-Age LFW benchmark |
| `cfp_ff.bin` | CFP Frontal-Frontal benchmark |
| `cfp_fp.bin` | CFP Frontal-Profile (pose) benchmark |
| `cplfw.bin` | Cross-Pose LFW benchmark |

---

## Configuration

Open `config.py` and check these lines near the bottom:

```python
RECORDIO_DATASET_PATH = r"C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112"
```

Change this path if your dataset is stored in a different folder.

---

## Training with RecordIO (Step-by-Step)

### Step 1 — Activate your virtual environment

Open PowerShell in the project folder and run:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2 — Verify the Dataset Loader (Quick Test Without Training)

Run this command to test the RecordIO loader stream without training the model:

```powershell
python train.py --dataset "C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112" --max-identities 100 --test-loader
```

This verifies that:
- RecordIO format is automatically detected (`train.rec` and `train.idx`).
- Batches can be lazily drawn from `train.rec` without memory spikes.
- Batch shapes and one-hot labels match expected dimensions.

### Step 3 — Verify a Single Training Step (Forward + Backward Pass Test)

Run this command to verify that model forward pass, loss calculation, GradientTape, optimizer backprop, and prototype memory updates all succeed on 1 real batch:

```powershell
python train.py --dataset "C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112" --max-identities 100 --test-step
```

### Step 4 — Quick experiment (100 identities, 5 epochs)

This is the recommended first run to make sure everything works:

```powershell
python train.py --recordio --max-identities 100 --epochs 5 --no-eval
```

What this does:
- `--recordio` : reads train.rec directly (no folder extraction needed)
- `--max-identities 100` : uses only the 100 identities with the most images
- `--epochs 5` : trains for 5 epochs (fast, for testing)
- `--no-eval` : skips benchmark evaluation (faster)

### Step 3 — Full experiment with evaluation

```powershell
python train.py --recordio --max-identities 100 --epochs 30
```

This trains for 30 epochs and then evaluates on all 6 benchmarks (lfw, agedb_30,
calfw, cfp_ff, cfp_fp, cplfw). Results are saved to `reports/`.

### Step 4 — Override dataset path

If your dataset is stored somewhere else:

```powershell
python train.py --recordio --dataset "D:\datasets\faces_webface_112x112" --max-identities 100
```

### Step 5 — Separate benchmark directory

If your .bin files are in a different location:

```powershell
python train.py --recordio --dataset "D:\data\casia" --benchmark-dir "D:\data\benchmarks"
```

---

## What Happens During Training (Plain English)

```
train.lst → identity index (who is in which record)
    ↓
train.idx → byte-offset map (where each record is in train.rec)
    ↓
Select top-100 identities (those with the most images)
    ↓
Identity-disjoint split: 70% train | 15% val | 15% test
(no person appears in more than one split)
    ↓
tf.data.Dataset reads images lazily from train.rec
(no images are extracted to folders)
    ↓
QualityAwareTrainer runs:
  - AdaptiveMarginLoss (ArcFace + per-sample margins)
  - PrototypeMemory updates (one prototype per identity)
  - Hard/noisy sample weighting
    ↓
After training: evaluate on lfw.bin, agedb_30.bin, etc.
Reports: AUC, EER, TAR@FAR → saved to reports/
```

---

## Outputs After Training

| File | Description |
|---|---|
| `models/embedding_model.h5` | Trained embedding model weights |
| `models/prototypes.npy` | Learned class prototype vectors |
| `results/logs/training_log.csv` | Loss and accuracy per epoch |
| `results/plots/` | Loss / accuracy curves |
| `reports/roc_lfw.png` | ROC curve for LFW benchmark |
| `reports/report_lfw.txt` | AUC, EER, TAR@FAR for LFW |
| *(same for agedb_30, calfw, cfp_ff, cfp_fp, cplfw)* | |

---

## Running Unit Tests

```powershell
# Run all tests (including RecordIO loader tests)
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

# Run only RecordIO tests
.venv\Scripts\python.exe -m unittest tests.test_recordio_loader -v
```

Expected output: **48 tests, 0 failures, 5 skipped**
(the 5 skipped tests are live-dataset tests that only run when
`CASIA_REC_PATH` environment variable is set to the dataset directory)

To run live tests:

```powershell
$env:CASIA_REC_PATH = "C:\Users\ansar\OneDrive\Desktop\faces_webface_112x112"
.venv\Scripts\python.exe -m unittest tests.test_recordio_loader -v
```

---

## Identity-Disjoint Split (No Data Leakage)

The most important data hygiene principle in face recognition:

> **A person used during training must NEVER appear in the evaluation set.**

This is enforced automatically. The split works at the **identity level**:

- Training identities: used to learn face embeddings
- Validation identities: used for early stopping (different people)
- Test identities: used for ROC / AUC evaluation (completely unseen people)

This prevents the model from "memorising" evaluation subjects, which would
give falsely optimistic accuracy numbers.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: train.rec` | Check `RECORDIO_DATASET_PATH` in config.py |
| Training is very slow | Reduce `--max-identities` or `--epochs` |
| Out of memory | Reduce `BATCH_SIZE` in config.py |
| Live tests skipped | Set `CASIA_REC_PATH` environment variable |
| benchmark .bin file not found | Make sure lfw.bin etc. are in the same folder as train.rec |

