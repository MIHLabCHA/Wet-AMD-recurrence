# WetAMD Recurrence Classification

This repository contains code for classifying recurrence of Wet AMD within 12 months after treatment using fundus(or OCT) images.
The model is trained using 5-fold cross-validation with PyTorch and NNI, and supports various backbones from the `timm` library.

---

## Overview

- **Task**: Binary classification (0 = No recurrence, 1 = Recurrence)
- **Input**: Fundus(or OCT) images (PNG format)
- **Backbones**: inception_v3, efficientnet_b0 etc.
- **Cross-validation**: 5-fold
- **Framework**: PyTorch + timm + NNI
- **Output**: Per-fold best models, metrics, loss/CM plots, final summary

---

## Code Description

### 1. Data Structure

Each fold must contain the following:
```plaintext
/fold_1/
├── train/
│   ├── class_0/
│   └── class_1/
└── val/
    ├── class_0/
    └── class_1/

- Images: PNG files.
- Only files matching a specified pattern (e.g., `'fun-002'`) are used.

### 2. Main Script

- Loads fold-wise datasets using a custom `Dataset` class
- Initializes model from `timm` with `num_classes=2`
- Trains per fold using cross-entropy loss and Adam optimizer
- Computes AUC 
- Saves:
  - Best model per fold (`.pth`)
  - Loss plot and confusion matrix (`.png`)
  - Fold metrics and final summary (`.pkl`, `.txt`)
- For dual-modality and multi-modality models, please refer to `01_Code/utils/model.py` for implementation details.


### 3. NNI Integration

NNI is used to search hyperparameters:

- `lr`
- `batchsize`
- `FE_model`
- `max_files_per_subject`

Search space and configuration must be defined in `search_space.json` and `config.yml`. The script uses `nni.get_next_parameter()` internally.

---

## Requirements

- Python 3.9+
- PyTorch (with appropriate CUDA)
- torchvision, timm, nni, scikit-learn, pandas, matplotlib, pillow

---

## Citation

This codebase is part of the work described in the following study:

**Uncertainty-Aware Selective Prediction of Neovascular Age-Related Macular Degeneration Recurrence Using Artificial Intelligence**

If you use this code or build upon it, please cite the above paper.  
(*Formal citation will be updated upon publication*)



