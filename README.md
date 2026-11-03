# MultiEyeNet: A Backbone-Diverse Deep Learning Ensemble for Ocular Disease Classification

**A rigorous parallel cross-modality benchmarking framework for automated multi-class screening across anterior-segment and fundus imaging, emphasizing diagnostic safety, probabilistic calibration, and domain-shift robustness.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aliyar4061/Multieyenet2/blob/main/colab_setup.ipynb)

---

## 📝 Abstract

**Objectives:** To develop and evaluate a parallel cross-modality deep learning benchmarking framework for multi-class ocular disease screening across anterior-segment and fundus imaging, with a critical emphasis on architectural robustness, probabilistic calibration, and diagnostic safety.

**Methods:** Eleven distinct architectural configurations, derived from five state-of-the-art backbones (DenseNet-121, EfficientNet-B2, ResNet-50, ViT-Base, Swin-Tiny), were evaluated. The best-performing models were integrated into modality-specific ensembles optimized via Sequential Least SQuares Programming (SLSQP) to minimize validation log-loss. Performance was assessed via zero-shot external cohort validation, controlled architectural ablation of Squeeze-and-Excitation (SE) and Graph Convolutional Network (GCN) modules, and multi-level domain-shift stress tests.

**Results:** 
- **Anterior-Segment:** The SLSQP-optimized ensemble achieved **97.10% accuracy (AUC 0.9969)**. Ablation demonstrated that removing SE/GCN modules from baseline architectures often reduced optimization overhead and improved accuracy.
- **Fundus:** The ensemble achieved **95.27% accuracy (AUC 0.9947)**. Convolutional architectures consistently outperformed data-hungry vision transformers in this data-constrained regime.
- **Vulnerability to Domain Shift:** While resilient to basic photometric variations, both modalities exhibited a catastrophic vulnerability to Gaussian noise, collapsing to **17.76% (Anterior)** and **26.10% (Fundus)** accuracy, dictating the mandatory integration of targeted noise augmentations prior to clinical deployment.
- **Diagnostic Safety:** Granular analysis revealed clinically significant False Negative Rates (FNR) in high-risk conditions: **Conjunctivitis (9.31%)** and **Glaucoma (10.02%)**, emphasizing that aggregate accuracy masks critical edge-case failures.

---

## 🚀 Key Features

- ✅ **Parallel Cross-Modality Benchmarking** – Independent evaluation pipelines for 5-class anterior and 4-class fundus datasets.
- ✅ **SLSQP-Optimized Ensemble** – Mathematically constrained weight fusion of top-3 backbones to minimize log-loss and improve probabilistic calibration.
- ✅ **Zero-Shot External Validation** – Evaluated against entirely unseen real-world external cohorts (ODIR-5K and `mohamedsayed12`).
- ✅ **Cryptographic Data Security** – Enforced a global **MD5 Hashing Protocol** across all splits to definitively eliminate exact/near-duplicates and prevent patient-level leakage.
- ✅ **Advanced Statistical Rigor** – Performance validated using McNemar's Test, DeLong's Permutation Test, Cohen's $d$ effect size, and Expected Calibration Error (ECE).
- ✅ **Per-Class Safety Analysis** – Explicit reporting of False Negative Rates (FNR) with bootstrapped 95% Confidence Intervals for clinical transparency.

---

## 📊 Dataset Summary

| Modality | Total Samples | Classes / Conditions | External Cohort |
|----------|---------------|----------------------|-----------------|
| **Anterior-Segment** | 2,297 | 5 (Normal, Cataract, Conjunctivitis, Uveitis, Eyelid Drooping) | `mohamedsayed12` subset |
| **Fundus** | 4,215 | 4 (Normal, Cataract, Diabetic Retinopathy, Glaucoma) | ODIR-5K subset |

*Note: Internal datasets were partitioned via stratified sampling (70% Train, 15% Validation, 15% Test) after MD5 deduplication.*

---

## 📈 Performance Highlights (TTA Ensemble)

| Metric | Anterior-Segment Ensemble | Fundus Ensemble |
|--------|---------------------------|-----------------|
| **Accuracy (95% CI)** | 97.10% (95.07 - 98.84) | 95.27% (93.68 - 96.84) |
| **AUC (Macro)** | 0.9969 | 0.9947 |
| **Expected Calib. Error (ECE)** | 0.1009 | 0.0669 |
| **Multi-Seed Acc (3 Seeds)** | 97.49% ± 0.36% | 94.63% ± 0.78% |
| **Critical FNR** | 9.31% (Conjunctivitis) | 10.02% (Glaucoma) |

---

## 🛠 Implementation Details

All experiments are implemented in **PyTorch 2.0+** using the `timm` library for pretrained backbone initialization.

### Training Hyperparameters & Regularization
| Component | Specification |
|-----------|---------------|
| **Input Resolution** | 224 × 224 pixels (bicubic interpolation) |
| **Batch Size** | 64 |
| **Optimizer** | AdamW (lr = 5e-4, weight decay = 1e-2) |
| **Scheduler** | Linear Warm-up (3 epochs) + Cosine Annealing |
| **Loss Function** | Focal Loss (γ = 2.0) + Label Smoothing (ε = 0.1) |
| **Data Augmentation** | Flip (p=0.5), Rotation ±20° (p=0.7), Color Jitter ±0.2 (Hue ±0.1, p=0.6), Coarse Dropout (max holes=8), CLAHE (p=0.3), Elastic Transform (p=0.1) |
| **Regularization** | MixUp (α=0.2, p=0.5) |
| **Test-Time Augmentation**| K=5 stochastic transformations (flips, rotation ±15°) |
| **Ensemble Fusion** | SLSQP optimization (minimizing Log-Loss subject to $\sum w_i = 1$) |

---

## 💻 Environment & Hardware

### System Requirements
- **OS:** Linux (Ubuntu 20.04+), Windows 10/11, or macOS (CPU-only mode)  
- **Python:** 3.8 – 3.10  
- **CUDA:** 11.7 or 11.8 (required for GPU training)  
- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) minimum; RTX 3090 / A100 recommended  
- **RAM:** 32 GB or higher  

### Software Dependencies
All required packages are listed in `requirements.txt`. Core libraries include:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `timm>=0.9.0` (ImageNet pretrained backbones)
- `albumentations>=1.3.0` (Advanced augmentations)
- `scikit-learn`, `pandas`, `numpy`, `scipy` (SLSQP and statistical tests)
- `matplotlib`, `seaborn`, `tqdm`
- `opencv-python-headless`, `pillow`

---

## 🔧 Installation & Usage

Clone the repository and install dependencies:

```bash
git clone [https://github.com/Aliyar4061/Multieyenet2.git](https://github.com/Aliyar4061/Multieyenet2.git)
cd Multieyenet2
pip install -r requirements.txt
