# 🌾 TerraSpectra: Hyperspectral Crop Disease Forecasting

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Data%20Science%20%26%20Machine%20Learning-green.svg)]()
[![Internship](https://img.shields.io/badge/Infotact%20Solutions-Internship%20Project-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 📌 Project Overview

**TerraSpectra** is an end-to-end Machine Learning and Remote Sensing framework engineered to detect, classify, and forecast crop diseases before visible symptoms manifest to the human eye. 

Unlike standard digital cameras (which only capture broad Red, Green, and Blue bands), **Hyperspectral Imaging (HSI)** records hundreds of continuous, narrow spectral bands across the **VNIR (Visible & Near-Infrared: 400–1000 nm)** and **SWIR (Short-Wave Infrared: 1000–2500 nm)** spectrum. 

By analyzing microscopic changes in leaf chlorophyll content, water absorption, and cellular scattering, **TerraSpectra** identifies pre-symptomatic physiological stress in plants, enabling early intervention, localized fungicide applications, and significant reductions in crop loss.

---

## 👨‍💻 Project Metadata & Internship Details

| Parameter | Details |
| :--- | :--- |
| **Intern Name** | **Vishal Yadav** |
| **Role** | Data Associate L1. Intern |
| **Domain** | Data Science & Machine Learning |
| **Organization** | **Infotact Solutions** (Electronic City, Bengaluru) |
| **Reference No.** | `ITS/DSML/1590` |
| **Internship Duration** | 3 Months |
| **Project Timeline** | **10 November 2026 – 10 December 2026** (1 Month) |
| **Email** | `vyadav276203@gmail.com` |

---

## 🔬 Core Science: Why Hyperspectral Imaging?

```
[ Incident Solar Radiation ]
             │
             ▼
    ┌─────────────────┐
    │  Plant Foliage  │
    └────────┬────────┘
             │
    ┌────────┴───────────────────────────┐
    ▼                                    ▼
[ Healthy Leaf ]                  [ Diseased / Stressed Leaf ]
- High Chlorophyll Absorption     - Chlorophyll Degradation (Red reflectance ↑)
  (Low Red Reflectance ~670nm)    - Internal Cell Collapse (NIR reflectance ↓)
- Strong Cellular Scattering      - Blue-shift of the "Red Edge" (700-740nm)
  (High NIR Reflectance ~800nm)   - Altered Xanthophyll Cycle (PRI shift ~531nm)
```

By computing narrow-band **Spectral Vegetation Indices**:
- **NDRE (Normalized Difference Red Edge):** Detects early chlorosis before necrosis occurs.
- **PRI (Photochemical Reflectance Index):** Monitors photosynthetic efficiency changes.
- **PSRI (Plant Senescence Reflectance Index):** Tracks cell breakdown and senescence.
- **NDVI / GNDVI:** Measures biomass vigor and canopy density.

---

## 📂 Project Directory Structure

```plaintext
TerraSpectra/
│
├── .gitignore                     # Git ignore rules for datasets, logs, virtualenvs
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Python package dependencies
├── app.py                         # Interactive Streamlit Web Demonstration
│
├── data/                          # Data store (ignored in git except .gitkeep)
│   ├── raw/                       # Original hyperspectral cubes (.mat, .hdr, .tif, .csv)
│   ├── processed/                 # Denoised, normalized, and calibrated spectral datasets
│   └── external/                  # Reference spectral signatures & benchmark databases
│
├── notebooks/                     # Exploratory Data Analysis & Prototyping
│   ├── README.md
│   ├── 01_eda_spectral_signatures.py
│   ├── 02_preprocessing_and_indices.py
│   └── 03_model_prototyping.py
│
├── src/                           # Modular production source code
│   ├── __init__.py
│   ├── data/                      # Ingestion & radiometric calibration
│   │   ├── __init__.py
│   │   ├── loader.py              # Multiformat dataset loader (.csv, .mat, .npy)
│   │   └── preprocess.py          # Savitzky-Golay filter, SNV, bad-band removal
│   │
│   ├── features/                  # Spectral feature engineering
│   │   ├── __init__.py
│   │   └── spectral_indices.py    # Narrow-band index computation (NDVI, NDRE, PRI, PSRI)
│   │
│   ├── models/                    # ML / DL pipelines
│   │   ├── __init__.py
│   │   ├── architectures.py       # RF, SVM, XGBoost, and 1D-CNN spectral pipelines
│   │   ├── train.py               # Model training, hyperparameter search, validation
│   │   └── predict.py             # Inference pipeline & disease probability scoring
│   │
│   └── utils/                     # Helper modules
│       ├── __init__.py
│       └── visualizer.py          # Spectral signature plotting, confusion matrix, ROC
│
├── models/                        # Serialized model weights & scalers (.pkl, .joblib, .pt)
├── reports/                       # Generated evaluation outputs & figures
│   └── figures/                   # Confusion matrix plots, spectral curve comparisons
│
└── docs/                          # Internship deliverables & documentation
    ├── project_report.md          # Comprehensive formal project report draft
    └── presentation_outline.md    # Presentation (PPT) slide-by-slide structure
```

---

## 🚀 Getting Started

### 1. Clone & Navigate
```bash
git clone https://github.com/vyadav276203/TerraSpectra.git
cd TerraSpectra
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Interactive Demo Dashboard
```bash
streamlit run app.py
```

---

## 🎯 Project Roadmap & Deliverables (Infotact Solutions)

As specified in the **Infotact Solutions Project Assignment Letter (`Ref: ITS/DSML/1590`)**:

| Phase | Duration | Scope & Deliverable | Status |
| :---: | :---: | :--- | :---: |
| **Phase 1** | Week 1 (10 Nov – 17 Nov) | Project Setup, Architecture Design, Dataset Acquisition & Literature Review | 🟡 In Progress |
| **Phase 2** | Week 2 (18 Nov – 24 Nov) | Exploratory Data Analysis (EDA), Radiometric Calibration, Spectral Index Extraction | ⚪ Pending |
| **Phase 3** | Week 3 (25 Nov – 01 Dec) | Model Training (RF, SVM, 1D-CNN), Cross-Validation & Metric Optimization | ⚪ Pending |
| **Phase 4** | Week 4 (02 Dec – 10 Dec) | Demonstration App, Final Report (`project_report.md`), PPT (`presentation_outline.md`), Code Review | ⚪ Pending |

### Required Submission Deliverables:
1. **Source Code:** Modular `src/` codebase with version control.
2. **Project Documentation:** Complete `README.md` and module docstrings.
3. **Project Report:** Documented methodology in `docs/project_report.md`.
4. **Presentation (PPT):** Slide structure ready in `docs/presentation_outline.md`.
5. **Model Evaluation Results:** Confusion matrices, F1-scores, accuracy curves.
6. **Final Project Demonstration:** Interactive web app via `streamlit run app.py`.

---

## 📊 Evaluation Metrics

Models are evaluated on balanced and imbalanced test splits using:
- **Accuracy & Balanced Accuracy**
- **Precision, Recall, Weighted F1-Score**
- **Cohen's Kappa ($\kappa$)** (Standard for remote sensing classification)
- **Confusion Matrix & ROC-AUC curves**

---

## 📜 Compliance & Confidentiality

This project is developed as part of an official internship assignment at **Infotact Solutions**. All methodologies, source files, and documents adhere to academic integrity, confidentiality, and anti-plagiarism guidelines.

