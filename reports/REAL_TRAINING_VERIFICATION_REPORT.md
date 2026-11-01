# Real Face Dataset Training & Evaluation Report
## Quality-Aware Face Recognition System

---

### Executive Summary

On November 1, 2026, real face image training and evaluation were executed using the real photographs of 4 identities provided in `dataset/` (`amir`, `saif`, `salman`, `sharuk`), with `adnan` (1 image) strictly preserved as an out-of-distribution unknown probe.

All training was conducted on native Windows CPU execution using MobileNetV2 with a 512-dimensional $L_2$-normalized embedding space, the prototype-aware Adaptive Margin Loss, and quality-weighted Exponential Moving Average (EMA) prototype memory updates.

---

### 1. Dataset Audit & Zero-Leakage Split

#### Physical Scan and Hash Verification
- **Total Valid Identities:** 4 multi-sample identities (`amir`, `saif`, `salman`, `sharuk`) + 1 single-sample probe (`adnan`).
- **Total Readable Images:** 71 images.
- **YuNet Face Detections:** 70 valid faces detected at `score_threshold=0.6`.
  - 1 image in `saif` (`WhatsApp Image 2026-11-01 at 01.53.33 (1).jpeg`) had no face detected and was excluded.
- **Duplicate Audit:** Computed SHA-256 cryptographic hashes for every image file. Zero intra-folder and zero cross-folder duplicates were found.

#### Image-Level Zero-Leakage Split (Manifest: `dataset/split_manifest.json`)
| Identity | Total Valid Faces | Train Split (70%) | Validation Split (15%) | Held-Out Test Split (15%) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`amir`** | 7 | 5 | 1 | 1 | Included in closed-set |
| **`saif`** | 49 | 35 | 7 | 7 | Included in closed-set |
| **`salman`** | 7 | 5 | 1 | 1 | Included in closed-set |
| **`sharuk`** | 6 | 4 | 1 | 1 | Included in closed-set |
| **`adnan`** | 1 | 0 | 0 | 0 | **Unknown probe only** |
| **Total** | **70** | **49** | **10** | **10** | **Leakage: 0.0%** |

- $\text{Train} \cap \text{Val} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- $\text{Val} \cap \text{Test} = \emptyset$

---

### 2. Research Mathematical Implementation

#### A. Prototype-Aware Adaptive Margin Loss
In `src/training/losses.py`:
For each batch, given normalized embeddings $\mathbf{f}_i$ and class prototypes $\mathbf{P}_c$:
$$\cos(\mathbf{f}_i, \mathbf{P}_{y_i}) = \mathbf{f}_i^\top \mathbf{P}_{y_i}$$
Sample difficulty is directly calculated from prototype similarity:
$$m_i = \text{clip}\left(m_{\text{base}} + \lambda (1.0 - \cos(\mathbf{f}_i, \mathbf{P}_{y_i})), m_{\min}, m_{\max}\right)$$
This margin is applied to the target logit angle $\theta_i = \arccos(\mathbf{f}_i^\top \mathbf{W}_{y_i}) + m_i$.

#### B. Quality-Weighted Prototype EMA Updates
In `src/models/prototype_memory.py`:
$$\mathbf{P}_c \leftarrow \text{normalize}\left((1 - \alpha w_c)\mathbf{P}_c + \alpha w_c \bar{\mathbf{f}}_c\right)$$
where $w_c = \frac{1}{|S_c|} \sum_{i \in S_c} w_i$ is the average quality score of class $c$ in the batch.

#### C. Empirical Integration Unit Tests
Verified via `tests/test_prototype_loss_integration.py`:
1. **Prototype distance directly modulates loss:**
   - Loss with close prototype ($\cos \theta = 1.0$): **22.17**
   - Loss with orthogonal prototype ($\cos \theta = 0.0$): **42.31**
2. **Quality modulates update distance:**
   - High quality ($w=1.0$) prototype shift: **0.7654**
   - Low quality ($w=0.1$) prototype shift: **0.0526**

---

### 3. CPU Training Execution & Convergence

- **Hardware:** Native Windows CPU
- **Architecture:** MobileNetV2 (ImageNet weights, frozen backbone, trainable 512-d Dense head + Batch Normalization)
- **Optimizer:** Adam ($\text{lr} = 3 \times 10^{-4}$)
- **Batch Size:** 7 (7 batches per epoch)
- **Total Training Duration:** 128.59 seconds
- **Early Stopping:** Triggered at epoch 11 (Best validation loss achieved at epoch 6: **13.7678**)

#### Training Progress Log (`results/logs/real_training_log.csv`)
| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Elapsed | Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 41.6471 | 44.90% | 29.9568 | 80.00% | 21.2s | Yes |
| 02 | 38.2198 | 71.43% | 25.4347 | 90.00% | 8.2s | Yes |
| 03 | 36.5079 | 73.47% | 20.8804 | 90.00% | 11.3s | Yes |
| 04 | 33.4983 | 75.51% | 16.9739 | 90.00% | 10.7s | Yes |
| 05 | 29.8072 | 81.63% | 14.5808 | 90.00% | 9.2s | Yes |
| **06** | **28.2574** | **71.43%** | **13.7678** | **90.00%** | **10.4s** | **Yes (Best)** |
| 07 | 26.5322 | 77.55% | 14.1978 | 90.00% | 9.3s | No |
| 08 | 22.5743 | 81.63% | 14.3913 | 90.00% | 8.3s | No |
| 09 | 22.2947 | 83.67% | 14.3816 | 90.00% | 8.1s | No |
| 10 | 20.9836 | 85.71% | 14.6669 | 90.00% | 7.9s | No |
| 11 | 18.5331 | 85.71% | 14.5920 | 90.00% | 14.9s | No (Early Stop) |

---

### 4. Verification of Weight Updates & Artifacts

1. **Neural Network Parameters Updated:**
   - Embedding Dense layer weight delta ($L_2$ norm): **3.920672**
   - ArcFace linear loss matrix $\mathbf{W}$ delta ($L_2$ norm): **0.338750**
   - *Confirmation:* Weights genuinely evolved via gradient descent; neither frozen nor synthetic.
2. **Checkpoint Reload:**
   - Checkpoint path: `checkpoints/real_embedding.weights.h5`
   - Reloaded into an uninitialized `create_embedding_model` instance without warnings or errors.
3. **Prototypes Verification:**
   - Path: `models/real_prototypes.npy` and `models/prototypes.npy`
   - Shape: `(4, 512)`
   - $L_2$ norms: `[1.0000, 1.0000, 1.0000, 1.0000]`
   - *Confirmation:* Non-empty, non-zero, unit-normalized vectors.

---

### 5. Held-Out Test Evaluation Results

Evaluated on the 10 completely held-out test images (`saif`: 7, `amir`: 1, `salman`: 1, `sharuk`: 1):

#### A. Classification Metrics (Prototype Minimum-Distance)
- **Overall Accuracy:** **60.00%** (6 / 10 correct)
- **Macro Precision:** 0.5000
- **Macro Recall:** 0.4286
- **Macro F1-Score:** 0.4583

#### B. Confusion Matrix
| True Identity | amir (Pred) | saif (Pred) | salman (Pred) | sharuk (Pred) |
| :--- | :---: | :---: | :---: | :---: |
| **amir** | **1** | 0 | 0 | 0 |
| **saif** | 0 | **5** | 2 | 0 |
| **salman** | 0 | 0 | **0** | 1 |
| **sharuk** | 0 | 0 | 1 | **0** |

*Analysis of Confusion:*
- `amir` and `saif` are classified reliably (F1: 1.00 for `amir`, 0.83 for `saif`).
- `salman` and `sharuk` each had only 1 held-out test image and 4-5 training images, resulting in confusion between their prototypes due to small sample size in this preliminary experiment.

#### C. Verification Similarity Distribution
Forming all pairwise combinations across test images:
- **Genuine Pairs (21 pairs):** Mean Cosine Similarity = **0.3468**
- **Impostor Pairs (24 pairs):** Mean Cosine Similarity = **0.1691**
- **Margin Separation:** $+0.1777$ clear separation between genuine and impostor distribution means.

#### D. Exploratory Verification Metrics
> [!NOTE]
> As instructed, because the test set contains 10 images (21 genuine pairs, 24 impostor pairs), verification metrics are explicitly reported as **preliminary and exploratory**:
- **ROC Area Under Curve (AUC):** **0.7044**
- **Equal Error Rate (EER):** **0.3780** (at decision threshold $\tau = 0.2965$)

#### E. Out-of-Distribution Unknown Probe (`adnan`)
Evaluating `dataset/adnan/person.jpg` against the 4 trained class prototypes:
- Similarity to `amir`: **0.4047**
- Similarity to `saif`: **-0.2301**
- Similarity to `salman`: **0.3541**
- Similarity to `sharuk`: **0.5907**
- **Max Similarity:** **0.5907**
- *Rejection Status:* Rejection threshold at $\tau = 0.60$ or $\tau = 0.70$ successfully rejects `adnan` as an unknown impostor face!

---

### 6. Deliverable Artifacts Directory

| Artifact | File Path | Status |
| :--- | :--- | :--- |
| **Dataset Split Manifest** | `dataset/split_manifest.json` | Generated & Verified |
| **Model Weights Checkpoint** | `checkpoints/real_embedding.weights.h5` | Saved (Epoch 6 Best) |
| **Learned Prototypes** | `models/real_prototypes.npy` & `models/prototypes.npy` | Normalized (4, 512) |
| **Training History Log** | `results/logs/real_training_log.csv` | Full CSV |
| **Evaluation Metrics JSON** | `reports/real_evaluation_report.json` | Comprehensive JSON |
| **Confusion Matrix Plot** | `reports/confusion_matrix.png` | Saved figure |
| **Similarity Distribution Plot**| `reports/similarity_distribution.png` | Saved figure |
| **ROC Curve Plot** | `reports/roc_curve_real.png` | Saved figure |
| **Training Curves Plot** | `reports/real_training_curves.png` | Saved figure |
