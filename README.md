# Quality-Aware Face Recognition
### Adaptive Empirical Prototypes, Dynamic Angular Margins & Confidence-Weighted Verification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![TensorFlow 2.21](https://img.shields.io/badge/tensorflow-2.21-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io)
[![CPU-Compatible](https://img.shields.io/badge/hardware-CPU%20native-green.svg)]()
[![Tests](https://img.shields.io/badge/Unit%20Tests-76%20Passed-brightgreen.svg)]()

---

## 1. Project Overview

**Quality-Aware Recognition** is a machine learning and computer vision framework for robust open-set face recognition and verification under uncontrolled, real-world imaging conditions. While traditional deep face recognition architectures (e.g., standard ArcFace, CosFace) assume uniform high quality and learn fixed identity weights via standard backpropagation, practical surveillance and mobile camera inputs routinely suffer from motion blur, suboptimal illumination, sensor noise, scale variations, and partial occlusions.

This project addresses these challenges by integrating **multi-metric image quality assessment**, **empirical class prototype memory**, **sample difficulty-adaptive angular margins**, and **confidence-weighted gradient backpropagation**.

---

## 2. Problem Statement

Standard angular margin loss functions (such as fixed-margin ArcFace) enforce a uniform angular penalty across all samples. This creates severe failure modes in real-world deployments:
1. **Noisy/Degraded Samples:** Blurry, poorly lit, or occluded face images are forced to satisfy the same aggressive geometric margin as pristine frontal captures, pulling the learned class representation away from the true identity cluster and corrupting model weights.
2. **Static Weight Matrices:** Linear classification weight vectors $W_c$ lack explicit empirical grounding and cannot easily adapt dynamically or support fast open-set enrollment without complete retraining.
3. **Overfitting to Easy Samples:** Clear, high-confidence samples receive unnecessary gradient updates while hard borderline cases lack differentiated margin attention.

---

## 3. Motivation

Face recognition systems deployed on mobile devices, low-power edge nodes, and college environments operate under strict constraints:
- **Zero or limited GPU resources (CPU execution required).**
- **Unconstrained capture environments with fluctuating image quality.**
- **Need for transparent, interpretable confidence estimation rather than overconfident false matches.**

By incorporating quality awareness directly into prototype updates and loss computation, the recognition system becomes robust, mathematically stable, and resilient to sensor noise.

---

## 4. Official Project Objectives

The project is architected to address five core research objectives:

- **Objective 1:** Learn adaptive empirical prototypes that accurately represent class centers for face recognition on the unit hypersphere.
- **Objective 2:** Update prototypes dynamically using feature similarity and quality weighting during model training via an Exponential Moving Average (EMA) formulation.
- **Objective 3:** Assign adaptive margins dynamically based on sample difficulty and prediction confidence ($m_i = \text{clip}(m_{\text{base}} + \lambda(1 - \cos(\theta_i)), m_{\min}, m_{\max})$).
- **Objective 4:** Improve recognition accuracy and stability by reducing the impact of hard, noisy, and degraded samples via quality-aware loss attenuation.
- **Objective 5:** Enhance robust face verification across pose, illumination, and quality variations with comprehensive ROC, AUC, EER, and TAR@FAR evaluation.

---

## 5. Proposed Methodology

```
                           INPUT FACE IMAGE
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │    YuNet DNN Face Detection   │ ──► Bounding Box & 5 Landmarks
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │    Multi-Metric Face Quality  │ ──► Sharpness, Brightness,
                   │         Assessment            │     Contrast, Size, Uniformity
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │      MobileNetV2 Embedding    │ ──► 512-d L2-Normalized Vector
                   │             Head              │
                   └──────────────┬────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
   ┌───────────────────────────┐     ┌───────────────────────────┐
   │ Training: Adaptive Margin │     │ Inference: Dynamic        │
   │ Loss & EMA Prototype Mem  │     │ Threshold Verification    │
   └───────────────────────────┘     └───────────────────────────┘
```

1. **Detection & Alignment:** OpenCV YuNet detects face boundaries, confidence, and 5 facial landmarks (eyes, nose, mouth corners).
2. **Quality Scoring:** Face crops are analyzed across Laplacian sharpness variance, optimal brightness range (100–170), pixel standard deviation contrast, resolution, and left/right illumination symmetry.
3. **Feature Embedding:** MobileNetV2 extracts a 512-dimensional vector projected onto the unit hypersphere via L2 normalization ($\|\mathbf{f}\|_2 = 1$).
4. **Adaptive Margin Loss:** Calculates angular distance $\theta_i$ to the ground-truth prototype and assigns a difficulty-dependent margin.
5. **Quality-Weighted Prototype Update:** High-quality samples strongly guide prototype drift; low-quality noisy samples are downweighted.

---

## 6. System Architecture & Logical Modules

The codebase is structured into modular components:

```
quality-aware-face-recognition/
├── app.py                          # Streamlit 7-section web application
├── config.py                       # Central configuration hyperparameters
├── train.py                        # Full training entry point
├── train_demo.py                   # Fast CPU demonstration training script
├── requirements.txt                # Windows CPU dependency specifications
├── README.md                       # Comprehensive project documentation
├── PROJECT_REPORT.md               # Evaluator college academic report
│
├── src/
│   ├── database.py                 # Isolated model-specific face databases
│   ├── face_model.py               # MobileNetV2 embedding & classifier factories
│   ├── image_processing.py         # Image loading & OpenCV Haar fallbacks
│   │
│   ├── models/
│   │   ├── face_model.py           # Model factory re-exports
│   │   ├── embedding_model.py      # Embedding model accessor
│   │   └── prototype_memory.py     # Objective 1 & 2: Empirical Prototype Memory
│   │
│   ├── training/
│   │   ├── losses.py               # Objective 3 & 4: Adaptive Margin Loss
│   │   ├── trainer.py              # Quality-aware training loop on GradientTape
│   │   └── train.py                # Training routines
│   │
│   ├── preprocessing/
│   │   ├── face_detection.py       # YuNet ONNX face & landmark detector
│   │   ├── image_preprocessing.py  # Normalization & quality degradation simulators
│   │   ├── quality_assessment.py   # Multi-metric FaceQualityAssessor
│   │   ├── dataset_loader.py       # Folder-based dataset scanning & disjoint split
│   │   └── recordio_loader.py      # MXNet RecordIO CASIA-WebFace loader
│   │
│   ├── recognition/
│   │   ├── recognizer.py           # QualityAwareRecognizer (1:N matching)
│   │   └── verification.py         # 1:1 Face Verification & joint quality logic
│   │
│   ├── evaluation/
│   │   └── evaluate.py             # ROC, AUC, EER, Accuracy, Precision, Recall, F1
│   │
│   └── utils/
│       ├── metrics.py              # Cosine similarity & threshold utilities
│       └── seed_utils.py           # Global deterministic seed management
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx    # 232 KB face detector
│   ├── face_recognition_sface_2021dec.onnx  # 38 MB lightweight demo engine
│   └── README.md
│
├── checkpoints/                    # Saved model weight checkpoints
├── data/                           # Serialized databases (.pkl)
├── reports/                        # Evaluation plots (ROC) and JSON metrics
├── results/                        # Training loss logs
│
└── tests/                          # 76 Unit tests (All Passing)
    ├── test_prototype_memory.py    # Prototype memory tests
    ├── test_adaptive_loss.py       # Adaptive margin loss tests
    ├── test_preprocessing.py       # Detection & quality tests
    ├── test_trainer_step.py        # GradientTape step tests
    ├── test_checkpoint_saving.py   # Weight & prototype persistence tests
    ├── test_evaluation.py          # ROC & verification metric tests
    └── test_recordio_loader.py     # RecordIO parsing tests
```

---

## 7. Mathematical Formulations

### A. Adaptive Empirical Prototypes (Objectives 1 & 2)

Each identity $c \in \{1, \dots, C\}$ is represented by an empirical prototype vector $\mathbf{P}_c \in \mathbb{R}^{512}$ with $\|\mathbf{P}_c\|_2 = 1$. Prototypes are initialized as the normalized class mean of early batch embeddings:
$$\mathbf{P}_c^{(0)} = \frac{\sum_{i \in \mathcal{S}_c} \mathbf{f}_i}{\left\|\sum_{i \in \mathcal{S}_c} \mathbf{f}_i\right\|_2}$$

During training, as mini-batch embeddings $\mathbf{f}_i$ arrive, prototypes update dynamically via a quality-weighted Exponential Moving Average (EMA):
$$\mathbf{P}_c \leftarrow \frac{(1 - \alpha \cdot \bar{w}_c)\mathbf{P}_c + \alpha \cdot \bar{w}_c \left(\frac{\sum_{i \in \mathcal{B}_c} w_i \mathbf{f}_i}{\sum_{i \in \mathcal{B}_c} w_i}\right)}{\left\|(1 - \alpha \cdot \bar{w}_c)\mathbf{P}_c + \alpha \cdot \bar{w}_c \left(\frac{\sum_{i \in \mathcal{B}_c} w_i \mathbf{f}_i}{\sum_{i \in \mathcal{B}_c} w_i}\right)\right\|_2}$$
where $\alpha \in (0, 1)$ is the prototype momentum and $w_i \in [0, 1]$ is the sample quality weight.

### B. Adaptive Angular Margin Loss (Objective 3)

Standard ArcFace introduces a constant angular margin $m$. In our system, the margin dynamically expands for hard samples and contracts for easy ones based on cosine similarity $\cos(\theta_i) = \mathbf{f}_i^\top \mathbf{P}_{y_i}$:
$$m_i = \text{clip}\left(m_{\text{base}} + \lambda (1 - \cos(\theta_i)), m_{\min}, m_{\max}\right)$$
The logits entering softmax are defined as:
$$\tilde{s}_{i, j} = \begin{cases} s \cdot \cos(\theta_i + m_i), & \text{if } j = y_i \\ s \cdot \cos(\theta_{i, j}), & \text{if } j \neq y_i \end{cases}$$
where $s = 30.0$ or $64.0$ is the hypersphere radius scale factor.

### C. Quality-Aware Confidence Weighting (Objective 4)

To prevent severe image degradation from injecting destabilizing gradients, sample loss contributions are modulated by:
$$w_i = \max\left(\text{floor}, \alpha_q Q_i + \alpha_c C_i\right)$$
$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^B w_i \cdot \left( -\log \frac{e^{\tilde{s}_{i, y_i}}}{\sum_{j=1}^C e^{\tilde{s}_{i, j}}} \right)$$
where $Q_i$ is the composite quality score and $C_i$ is the Softmax prediction confidence.

---

## 8. Dataset Information

### CASIA-WebFace Configuration
- **Status:** Not pre-downloaded (Honest Demonstration Policy). Full training on CASIA-WebFace is designed for multi-hour GPU runs.
- **RecordIO Structure:** Compatible with MXNet binary RecordIO packages (`train.rec`, `train.idx`, `train.lst`).
- **Setup for Full CASIA-WebFace Training:**
  1. Download `faces_webface_112x112` from the CASIA-WebFace repository.
  2. Place files into `dataset/casia_webface/` or edit `RECORDIO_DATASET_PATH` in `config.py`.
  3. Run: `python train.py --recordio`

### CPU Demonstration Mode
- For evaluation on CPU without downloading large datasets, the project includes an automated synthetic in-memory multi-identity generator that exercises the entire training loop, gradient tape, prototype updates, and verification pipeline in ~15 seconds:
  ```bash
  python train_demo.py
  ```

---

## 9. Model Architecture & Dual-Engine Consistency

| Feature | Research Engine (`MobileNetV2-512`) | Demo Engine (`SFace-128`) |
| :--- | :--- | :--- |
| **Backbone** | MobileNetV2 (ImageNet weights) | SFace CNN (OpenCV Zoo) |
| **Embedding Dimension** | **512-dimensional** | **128-dimensional** |
| **Normalization** | L2 Hyperspherical Unit Norm | L2 Unit Norm |
| **Prototype Method** | Dynamic Quality-Weighted EMA | Empirical Center Mean |
| **Database Storage** | `data/database_mobilenet_512.pkl` | `data/database_sface_128.pkl` |
| **Cross-Mixing** | **Strictly Prohibited & Isolated** | **Strictly Prohibited & Isolated** |

---

## 10. Windows Installation and Execution

### Step 1: Create Virtual Environment
```powershell
python -m venv .venv
```

### Step 2: Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Unit Tests (76 Tests)
```powershell
python -m unittest discover tests
```

### Step 5: Run Fast CPU Demonstration Training
```powershell
python train_demo.py
```

### Step 6: Launch Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Application Features

The modern Streamlit interface comprises 7 dedicated sections:

1. **🏠 Home / Dashboard:** Live CPU/RAM system diagnostics, status of dataset and pretrained weights, active engine selector, and research framework breakdown.
2. **👤 Face Registration:** Multi-image enrollment with automatic face bounding box, 5-point landmark visualizer, quality score breakdown, and L2-normalized prototype calculation.
3. **📸 Face Recognition:** 1:N identity classification with quality-adaptive decision boundaries (RECOGNIZED, UNKNOWN, or UNRELIABLE LOW-QUALITY WARNING), progress bars, and similarity breakdown.
4. **👥 Face Verification:** 1:1 pairwise matching displaying joint pair quality $Q_{\text{pair}} = \sqrt{Q_1 \cdot Q_2}$, side-by-side cropped comparisons, and adaptive decision thresholding.
5. **🔍 Quality Assessment & Simulator:** Interactive degradation controls (Gaussian blur, underexposure, dynamic range reduction, Gaussian sensor noise) showing real-time impact on quality metrics and recognition confidence.
6. **🔬 Research & Prototype Dashboard:** Interactive 2D hypersphere prototype update simulator, adaptive margin curve visualizer ($\Delta m$ vs $\cos(\theta)$), quality-confidence weight landscape, and live training loss logs.
7. **📊 Evaluation & Benchmarking:** Quantitative evaluation computing Accuracy, Precision, Recall, Macro F1-Score, Receiver Operating Characteristic (ROC) curve, AUC, Equal Error Rate (EER), and TAR at FAR=0.1%, 1%, 10%.

---

## 12. Limitations

1. **Hardware Scope:** Native Windows CPU execution without GPU acceleration limits training throughput on multi-gigabyte datasets (e.g. CASIA-WebFace).
2. **Extreme Head Poses:** Profile faces exceeding 60° yaw reduce landmark localization confidence.
3. **Severe Resolution Loss:** Face regions smaller than 40×40 pixels trigger the low-quality safety reject threshold to avoid false positive classification.

---

## 13. Future Improvements

1. Implementation of Vision Transformer (ViT) backbones with self-attention for occlusion handling.
2. Integration of 3D facial mesh reconstruction (e.g. MediaPipe Mesh) for 68-landmark pose normalization.
3. DirectML / ONNX Runtime GPU acceleration for Windows laptops with integrated AMD/Intel graphics.

---

## 14. License

This academic project is distributed under the MIT License.
