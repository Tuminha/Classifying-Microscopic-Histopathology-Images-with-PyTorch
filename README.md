# 🔬 PatchCamelyon (PCam) — Binary Tumor Classification with a Simple CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/Status-Learning-green.svg)
![Domain](https://img.shields.io/badge/Domain-Medical_Imaging-purple.svg)

**Educational project: Build a CNN from scratch to classify tumor patches in histopathology images**

[🎯 Overview](#-purpose--learning-style) • [📊 Dataset](#-dataset) • [🚀 Quick-Start](#-environment) • [📦 Notebooks](#-project-roadmap-notebooks)

</div>

> First hands-on CNN for medical imaging: learning convolutions, pooling, augmentation, and binary classification with PyTorch — not production-ready, but honest learning work.

---

## 👨‍💻 Author
<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through CodeCademy • Building AI solutions step by step*

</div>

---

## 🎯 Purpose & Learning Style

This project teaches **binary tumor classification** on 96×96 RGB histopathology image patches using a **simple Convolutional Neural Network (CNN)** built from scratch in PyTorch.

### Pedagogical Approach
- **No complete solution code provided** — you learn by doing
- Every notebook follows: **Concept Primer → Objectives → Acceptance Criteria → Numbered TODO cells (with hints) → Reflection prompts**
- Explains **what/why/how** and **expected tensor shapes** before each TODO
- Consistent variable naming throughout (`train_transform`, `val_test_transform`, `train_dataloader`, `cnn_model`, etc.)
- Small PCam subset provided via CSV + images folder

---

## 📊 Dataset

**PatchCamelyon (PCam)** is a binary classification dataset derived from lymph node histopathology scans.

- **Image Size:** 96×96 RGB patches
- **Labels:** 
  - `0` = Normal (no tumor tissue)
  - `1` = Tumor (metastatic tissue present)
- **Source:** Small subset provided in `data/` directory
  - `train_labels.csv` → training images
  - `validation_labels.csv` → validation images
  - `test_labels.csv` → test images
  - `data/pcam_images/` → actual PNG files referenced by CSVs

---

## 🗺 Project Roadmap (Notebooks)

All notebooks are **instructional** — you write the code via TODOs and hints. Run them in order:

| Notebook | Focus | What You'll Build |
|----------|-------|-------------------|
| `00_overview.ipynb` | Pipeline Map | Understand the full ML pipeline from images → predictions |
| `01_transforms_train.ipynb` | Train Augmentations | Build `train_transform` with augmentation pipeline |
| `02_load_train_loader.ipynb` | Train Dataset & Loader | Instantiate `PCamDataset` and `DataLoader` for training |
| `03_transforms_val_test.ipynb` | Val/Test Transforms (No Aug) | Build deterministic `val_test_transform` |
| `04_load_val_test_loaders.ipynb` | Val/Test Loaders | Create validation and test data loaders |
| `05_simple_cnn_scaffold.ipynb` | CNN Architecture | Define `SimpleCNN` class (3 conv blocks + 2 FC layers) |
| `06_device_loss_opt.ipynb` | Device, Loss, Optimizer | Setup GPU/CPU, loss function, and optimizer |
| `07_train_validate_loops.ipynb` | Train 5 Epochs + Val Loss | Implement training and validation loops |
| `08_test_inference_metrics.ipynb` | Test & Metrics | Run inference on test set and generate classification report |
| `99_lab_notes.ipynb` | Learning Journal | Document your learning journey and experiments |

---

## 💻 Environment

### Prerequisites
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python ≥3.10
- PyTorch ≥2.0
- Torchvision
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Jupyter

### Setup
```bash
git clone https://github.com/Tuminha/microscopic_histopathology.git
cd microscopic_histopathology
jupyter notebook notebooks/00_overview.ipynb
```

---

## 🚀 Run Order & Deliverables

1. **Execute notebooks sequentially** (00 → 08)
2. **Fill in TODO cells** using hints provided
3. **Expected Deliverables:**
   - ✅ Training loss curve (decreasing over 5 epochs)
   - ✅ Validation loss per epoch (tracking generalization)
   - ✅ Test classification report (precision, recall, F1 for Normal/Tumor)
   - ✅ Personal reflections in `99_lab_notes.ipynb`

---

## 🛠 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Processing | Pandas, NumPy | Load CSVs and manage labels |
| Image Loading | PIL/Pillow (via PyTorch) | Read PNG images |
| Transforms | Torchvision | Augmentation & normalization |
| Deep Learning | PyTorch | CNN architecture, training, inference |
| Evaluation | Scikit-learn | Classification metrics |
| Visualization | Matplotlib | Loss curves and confusion matrices |
| Version Control | Git/GitHub | Track progress |

---

## 📝 Learning Journey

### 🎓 Learning Objectives
- Understand **data augmentation** for training vs deterministic transforms for evaluation
- Build a **CNN from scratch** with Conv2D → ReLU → MaxPool → Fully Connected layers
- Implement **training and validation loops** in PyTorch
- Use **BCELoss** with Sigmoid activation for binary classification
- Generate and interpret **classification reports** (precision/recall/F1)

### 🏆 Key Achievements
- [x] **Notebook 01**: Build training transform pipeline with augmentation (Resize → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize)
- [x] **Notebook 01**: Understand H&E staining, realistic augmentations for histopathology, and normalization mathematics
- [x] **Notebook 02**: Instantiate PCamDataset and DataLoader with proper batching and shuffling
- [x] **Notebook 02**: Grasp why shuffling prevents order-based overfitting and batch size impact on generalization
- [x] **Notebook 03**: Build deterministic validation/test transforms and understand evaluation consistency
- [x] **Notebook 04**: Create validation and test data loaders with proper evaluation settings
- [ ] Build a 3-layer CNN architecture
- [ ] Train model with GPU acceleration
- [ ] Evaluate model with proper metrics
- [ ] Experiment with hyperparameters (learning rate, batch size)
- [ ] Try alternative architectures (ResNet, EfficientNet)
- [ ] Implement class balancing strategies

---

## 📈 Current Progress

### ✅ Completed Notebooks
- **Notebook 01**: Training Transforms & Augmentation
  - Built `train_transform` with proper order: Resize → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize
  - Learned H&E staining basics and realistic augmentations for histopathology
  - Mastered normalization mathematics: `(0.8 - 0.5) / 0.5 = 0.6`
  - Verified tensor shapes: `[3, 96, 96]` and value ranges: `[-1, 1]`

- **Notebook 02**: Training Dataset & DataLoader
  - Fixed PCamDataset column mismatch (`filename` vs `image_id`)
  - Instantiated `train_dataset` with 601 training samples
  - Created `train_dataloader` with batch_size=8, shuffle=True
  - Verified batch shapes: `images=[8,3,96,96]`, `labels=[8]`
  - Understood shuffling prevents order-based overfitting

- **Notebook 03**: Validation/Test Transforms
  - Built deterministic `val_test_transform` (Resize → ToTensor → Normalize)
  - Verified transform determinism: same input → same output
  - Understood why validation/test must be deterministic (no random augmentation)
  - Grasped normalization consistency across train/val/test splits

- **Notebook 04**: Validation/Test DataLoaders
  - Created `val_dataloader` and `test_dataloader` with batch_size=32, shuffle=False
  - Verified batch shapes: `images=[32,3,96,96]`, `labels=[32]`
  - Understood why larger batch sizes work for evaluation (no gradients needed)
  - Grasped importance of shuffle=False for consistent validation results

### 🎯 Next Up
- **Notebook 05**: Simple CNN Architecture

---

**🚨 CRITICAL: This is an educational project, NOT a medical device.**

- This model is for **learning purposes only**
- **Never use model outputs for clinical decisions**
- Real medical AI systems require:
  - Rigorous validation on diverse patient populations
  - Regulatory approval (FDA, CE marking, etc.)
  - Clinical integration with physician oversight
  - Continuous monitoring for bias and drift
- **False Negatives** (missing tumors) have severe clinical consequences
- **False Positives** lead to unnecessary biopsies and patient anxiety

> *"With great ML power comes great responsibility"* — Always consider real-world impact.

---

## 🚀 Next Steps

**Immediate (Notebooks 05-08):**
- [ ] **Notebook 05**: Define SimpleCNN architecture (3 conv blocks + 2 FC layers)
- [ ] **Notebook 06**: Setup device, loss function, and optimizer
- [ ] **Notebook 07**: Implement training and validation loops
- [ ] **Notebook 08**: Run test inference and generate metrics

**Future Enhancements:**
- [ ] Experiment with deeper architectures (ResNet18, EfficientNet-B0)
- [ ] Implement learning rate scheduling
- [ ] Add dropout layers to reduce overfitting
- [ ] Visualize activation maps (Grad-CAM)
- [ ] Try class weighting for imbalanced datasets
- [ ] Compare BCELoss vs BCEWithLogitsLoss
- [ ] Implement early stopping based on validation loss

---

## 📚 Key Concepts Explained

### Transform Order
- **Correct:** Resize/Augmentation → ToTensor → Normalize
- **Why:** Color augmentation works on PIL images; normalization needs tensors

### Normalization (mean=0.5, std=0.5)
- Centers pixel values from [0,1] to [-1,1]
- Stabilizes gradients during training
- Must use **same normalization** for train/val/test

### Spatial Dimension Changes
- **Input:** 96×96
- **After Pool 1:** 48×48
- **After Pool 2:** 24×24
- **After Pool 3:** 12×12
- **Flattened:** 128 channels × 12 × 12 = 18,432 features

### Batch Sizes
- **Train:** 8 (smaller for memory + more gradient updates)
- **Eval:** 32 (larger for speed, no backward pass)

### Sigmoid + BCELoss
- **What we use:** Final Sigmoid layer + `BCELoss`
- **Alternative:** No Sigmoid + `BCEWithLogitsLoss` (more numerically stable)
- **Why we keep Sigmoid:** Easier to interpret outputs as probabilities [0,1]

---

## 📄 License
MIT License (see [LICENSE](LICENSE))

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**  
*Building AI solutions one patch at a time* 🔬🚀

</div>

