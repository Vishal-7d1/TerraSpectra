# 🌾 TerraSpectra: Hyperspectral Crop Disease Forecasting

[![Domain](https://img.shields.io/badge/Domain-Data%20Science%20%26%20Machine%20Learning-green.svg)]()
[![Language](https://img.shields.io/badge/Language-Python%203.10%2B-blue.svg)]()
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![Status](https://img.shields.io/badge/Status-Project%20Structure%20Initialized-orange.svg)]()

---

## 📖 1. Project Introduction

**TerraSpectra** is a specialized Data Science and Machine Learning project focused on agricultural remote sensing and plant pathology. Traditional optical RGB photography captures only broad visible light (Red, Green, Blue) and frequently fails to identify crop diseases until visual foliar lesions (chlorosis and necrosis) become prominent—by which point irreversible yield damage and pathogen propagation have occurred.

**Hyperspectral Imaging (HSI)** overcomes this limitation by capturing continuous, narrow spectral bands spanning the **Visible and Near-Infrared (VNIR: 400nm – 1000nm)** and **Short-Wave Infrared (SWIR: 1000nm – 2500nm)** spectrum. TerraSpectra aims to analyze these narrow-band spectral signatures to identify physiological and biochemical plant changes (such as chlorophyll breakdown, water stress, and cell wall degradation) for **early, pre-symptomatic crop disease detection and forecasting**.

---

## 🎯 2. Project Objective

The primary objective of **TerraSpectra** is to develop a modular, end-to-end Machine Learning pipeline that:
1. Ingests and standardizes hyperspectral crop reflectance data.
2. Performs radiometric calibration, noise reduction, and spectral baseline analysis.
3. Conducts rigorous Exploratory Data Analysis (EDA) on spectral signatures of healthy vs. infected crops.
4. Extracts domain-specific spectral vegetation indices (e.g., NDVI, NDRE, PRI, PSRI) and performs dimensionality reduction.
5. Designs, trains, benchmarks, and evaluates machine learning classifiers to predict disease categories and severity stages.
6. Builds a disease prediction system and an interactive frontend dashboard for agronomists and decision-makers.
7. Produces comprehensive project documentation, reports, and presentation deliverables.

---

## 🔭 3. Project Scope

- **Data Handling:** Collection, organization, and validation of real-world agricultural hyperspectral datasets (or standard benchmark cubes/spectral libraries).
- **Preprocessing:** Savitzky-Golay filtering, Standard Normal Variate (SNV) transformation, continuum removal, and removal of noisy atmospheric water absorption bands.
- **Spectral Analysis:** Quantitative evaluation of the "Red-Edge" shift (690nm–740nm) and NIR reflectance plateau dynamics.
- **Feature Engineering:** Calculation of narrow-band vegetation indices, Principal Component Analysis (PCA), and feature selection.
- **Machine Learning Modeling:** Supervised classification and staging using algorithms such as Random Forest, Support Vector Machines (SVM), and Gradient Boosting.
- **Diagnostic System:** Real-time inference pipeline translating raw spectral measurements into risk levels and actionable agronomic prescriptions.
- **Frontend Dashboard:** Interactive visualization tool built with Streamlit for spectral signature exploration and crop health diagnosis.

---

## 🛠️ 4. Technology Stack

- **Core Programming Language:** Python 3.10+
- **Data Manipulation & Scientific Computing:** NumPy, Pandas, SciPy
- **Hyperspectral Processing:** Spectral Python (SPy)
- **Machine Learning & Modeling:** Scikit-Learn
- **Data Visualization & Analytics:** Matplotlib, Seaborn
- **Interactive Prototyping:** Jupyter Notebook / JupyterLab
- **Frontend Application:** Streamlit
- **Version Control:** Git & GitHub

---

## 📂 5. Project Structure

The project follows a clean, modular, and sequential numbering hierarchy inspired by industry best practices:

```text
TerraSpectra/
│
├── .venv/                              # Python Virtual Environment
│
├── 01_Dataset_Collection/              # Data ingestion and raw/processed repository
│   ├── raw_data/                       # Unmodified, original hyperspectral datasets
│   ├── processed_data/                 # Cleaned and standardized spectral matrices
│   ├── dataset_analysis.py             # Dataset validation and integrity audit script
│   └── README.md                       # Module documentation
│
├── 02_Data_Preprocessing/              # Signal processing, filtering, and calibration
│   ├── data_cleaning.py                # Outlier detection and missing value handling
│   ├── normalization.py                # SNV and radiometric reflectance normalization
│   ├── preprocessing.py                # Main preprocessing pipeline (Savitzky-Golay, etc.)
│   └── README.md                       # Module documentation
│
├── 03_Exploratory_Data_Analysis/       # Spectral analytics and visualization
│   ├── eda.py                          # Statistical distributions and exploratory plots
│   ├── spectral_analysis.py            # Spectral reflectance curves and Red-Edge derivative analysis
│   ├── outputs/                        # Saved EDA figures, plots, and charts
│   └── README.md                       # Module documentation
│
├── 04_Feature_Engineering/             # Spectral index calculation & dimensionality reduction
│   ├── feature_extraction.py           # Narrow-band vegetation indices (NDVI, NDRE, PRI, PSRI)
│   ├── feature_selection.py            # PCA and band importance ranking
│   └── README.md                       # Module documentation
│
├── 05_Machine_Learning/                # Model development, cross-validation & evaluation
│   ├── train.py                        # Model training and hyperparameter optimization
│   ├── evaluate.py                     # Evaluation metrics, confusion matrix, ROC-AUC
│   ├── models/                         # Serialized model checkpoints (.joblib, .pkl)
│   ├── outputs/                        # Evaluation plots, benchmark comparisons, metric reports
│   └── README.md                       # Module documentation
│
├── 06_Prediction_System/               # Inference engine and risk assessment
│   ├── predict.py                      # Prediction service for incoming spectral vectors
│   └── README.md                       # Module documentation
│
├── 07_Frontend_Dashboard/              # User interface and interactive demo
│   ├── app.py                          # Streamlit web application
│   └── README.md                       # Module documentation
│
├── 08_Final_Deliverables/              # Formal documentation and submission files
│   ├── Project_Report/                 # Comprehensive project report
│   ├── PPT/                            # Presentation slides and review deck
│   ├── Documentation/                  # Technical system architecture and user guides
│   ├── Evaluation_Results/             # Summary of final model metrics and benchmarks
│   └── README.md                       # Module documentation
│
├── .gitignore                          # Git ignore rules
├── README.md                           # Master project documentation
└── requirements.txt                    # Project dependency specification
```

---

## 🔄 6. Development Workflow

The project is structured to proceed in sequential phases:

```
[ Phase 01: Dataset Collection ]
       │
       ▼
[ Phase 02: Data Preprocessing & Calibration ]
       │
       ▼
[ Phase 03: Exploratory & Spectral Analysis ]
       │
       ▼
[ Phase 04: Feature Engineering & PCA ]
       │
       ▼
[ Phase 05: Machine Learning Model Training ]
       │
       ▼
[ Phase 06: Disease Prediction Pipeline ]
       │
       ▼
[ Phase 07: Frontend Dashboard Deployment ]
       │
       ▼
[ Phase 08: Final Deliverables & Reporting ]
```

---

## 💻 7. Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/vyadav276203/TerraSpectra.git
cd TerraSpectra
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Initial Dependencies
```bash
pip install -r requirements.txt
```

---

## 📌 8. Project Status

- [x] Project architecture and folder hierarchy established.
- [x] Module-level documentation initialized.
- [x] Virtual environment and dependency specifications configured.
- [ ] Phase 1: Real dataset selection and acquisition (Pending next instructions).
- [ ] Phase 2: Data preprocessing and spectral filtering (Pending).
- [ ] Phase 3: Exploratory spectral curve analysis (Pending).
- [ ] Phase 4: Feature extraction & index computation (Pending).
- [ ] Phase 5: Machine Learning modeling and validation (Pending).
- [ ] Phase 6: Inference pipeline implementation (Pending).
- [ ] Phase 7: Streamlit frontend dashboard (Pending).
- [ ] Phase 8: Final report and presentation preparation (Pending).
