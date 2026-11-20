# 🌾 TerraSpectra: 04_Feature_Engineering Module

**Project:** TerraSpectra – Hyperspectral Crop Disease Forecasting  
**Organization:** Infotact Solutions  
**Domain:** Data Science & Machine Learning (Agricultural Hyperspectral Remote Sensing)  
**Pipeline Stage:** Module 04 – Scientific Feature Engineering, Leakage-Free Selection & Dimensionality Reduction  

---

## 📌 1. Executive Summary & Purpose

The **04_Feature_Engineering** module bridges preprocessed hyperspectral reflectance data (`02_Data_Preprocessing`) and predictive machine learning models (`05_Machine_Learning`). Contiguous hyperspectral data suffer from extreme multicollinearity and high dimensionality (653 contiguous spectral bands from 450.82 nm to 2396.54 nm), known as the **Hughes phenomenon** (curse of dimensionality). 

This module implements a mathematically rigorous, biologically informed, and leakage-free feature engineering architecture:
1. **Domain Feature Extraction:** Derives 51 high-value scientific features capturing chlorophyll degradation, xanthophyll cycle kinetics, cellular water attenuation, red-edge inflection shifts, and regional spectral moments.
2. **Biological Group Partitioning:** Enforces strict physical plant specimen isolation grouped by `barcode` (70% Train, 15% Validation, 15% Test) with **0% biological plant overlap**.
3. **Leakage-Free Feature Selection Cascade:** Fits variance filtering, correlation thresholding ($|r| < 0.95$), and Random Forest Gini importance **strictly on the training split**.
4. **Unsupervised Dimensionality Reduction:** Fits StandardScaler and Principal Component Analysis (PCA) strictly on the training partition and serializes the transformer (`fitted_pca.joblib`).
5. **Multi-Method Triangulation:** Validates feature rankings across Random Forest Gini Importance, Mutual Information (MI), and ANOVA F-statistic.
6. **Defensive Verification:** Fully tested via 19 automated pytest unit and integration tests (100% pass rate).

---

## 🔬 2. Mathematical Formulations of Engineered Features

The pipeline extracts **51 scientific features** across 4 distinct biophysical categories:

### A. Narrow-Band Vegetation & Physiological Indices (10 Features)

| Feature Name | Target Biophysical Mechanism | Mathematical Formula | Wavelength Targets ($\lambda$ in nm) |
| :--- | :--- | :--- | :--- |
| **`VI_NDVI`** | Canopy Greenness & General Biomass | $\frac{R_{800} - R_{680}}{R_{800} + R_{680}}$ | NIR: 800 nm, Red: 680 nm |
| **`VI_NDRE`** | Pre-Symptomatic Chlorophyll Attenuation | $\frac{R_{790} - R_{720}}{R_{790} + R_{720}}$ | NIR: 790 nm, Red-Edge: 720 nm |
| **`VI_PRI`** | Xanthophyll Light-Use Efficiency Stress | $\frac{R_{531} - R_{570}}{R_{531} + R_{570}}$ | Photochemical: 531 nm, Ref: 570 nm |
| **`VI_mSR705`** | Chlorophyll Pigment Density | $\frac{R_{750} - R_{450}}{R_{705} - R_{450}}$ | NIR: 750 nm, RE: 705 nm, Blue: 450 nm |
| **`VI_WBI`** | Canopy Cellular Hydration State | $\frac{R_{900}}{R_{970}}$ | NIR: 900 nm, Water Dip: 970 nm |
| **`VI_CRI`** | Carotenoid Pigment Concentration | $\left(\frac{1}{R_{510}}\right) - \left(\frac{1}{R_{550}}\right)$ | Carotenoid: 510 nm, Green: 550 nm |
| **`VI_PSRI`** | Plant Senescence & Cellular Necrosis | $\frac{R_{680} - R_{500}}{R_{750}}$ | Red: 680 nm, Carotenoid: 500 nm, NIR: 750 nm |
| **`VI_GNDVI`** | Green Canopy Vigor | $\frac{R_{800} - R_{550}}{R_{800} + R_{550}}$ | NIR: 800 nm, Green: 550 nm |
| **`VI_MCARI`** | Chlorophyll Absorption / Soil-Insensitive | $\left[(R_{700} - R_{670}) - 0.2(R_{700} - R_{550})\right] \cdot \left(\frac{R_{700}}{R_{670}}\right)$ | 700 nm, 670 nm, 550 nm |
| **`VI_SIPI`** | Carotenoid / Chlorophyll-a Ratio | $\frac{R_{800} - R_{450}}{R_{800} - R_{680}}$ | 800 nm, 450 nm, 680 nm |

### B. Spectral Derivative Dynamics & Red-Edge Position (8 Features)

Contiguous derivatives are computed using second-order accurate central differences accounting for non-uniform band spacing:
$$\frac{dR}{d\lambda}\Bigg|_i = \frac{R_{i+1} - R_{i-1}}{\lambda_{i+1} - \lambda_{i-1}}, \quad \frac{d^2R}{d\lambda^2}\Bigg|_i = \frac{d1_{i+1} - d1_{i-1}}{\lambda_{i+1} - \lambda_{i-1}}$$

- **`DERIV_REP_wavelength_nm`**: Red-Edge Position, $\lambda_{\text{REP}} = \arg\max_{\lambda \in [680, 750]} \frac{dR}{d\lambda}$ (shifts blue under disease-induced chlorosis).
- **`DERIV_REP_amplitude`**: Peak maximum slope amplitude $\max \frac{dR}{d\lambda}$.
- **`DERIV_REP_area`**: Trapezoidal integrated derivative area $\int_{680}^{750} \frac{dR}{d\lambda} d\lambda$.
- **`DERIV_green_slope`**: Average 1st derivative in green ascent window ($520 - 560\text{ nm}$).
- **`DERIV_swir_water_slope_1450`**: Average derivative slope entering primary water absorption band ($1400 - 1460\text{ nm}$).
- **`DERIV_swir_water_slope_1940`**: Average derivative slope entering secondary water band ($1900 - 1950\text{ nm}$).
- **`DERIV_red_edge_min_curvature`**: Minimum second derivative in red edge ($\min \frac{d^2R}{d\lambda^2}$).
- **`DERIV_red_edge_max_curvature`**: Maximum second derivative in red edge ($\max \frac{d^2R}{d\lambda^2}$).

### C. Spectral Shape & Continuum Water Absorption Depths (11 Features)

- **`SHAPE_total_spectral_auc`**: Total integrated area under full spectrum $\int_{450}^{2396} R(\lambda) d\lambda$.
- **`SHAPE_vis_slope`**: Visible ascent slope $\frac{R_{680} - R_{450}}{\lambda_{680} - \lambda_{450}}$.
- **`SHAPE_red_edge_slope`**: Red-edge transition slope $\frac{R_{750} - R_{680}}{\lambda_{750} - \lambda_{680}}$.
- **`SHAPE_nir_plateau_mean`**: Mean reflectance across structural NIR plateau ($750 - 1100\text{ nm}$).
- **`SHAPE_nir_plateau_slope`**: Slope of NIR plateau $\frac{R_{1100} - R_{750}}{\lambda_{1100} - \lambda_{750}}$.
- **`SHAPE_water_depth_1450`**: Continuum-relative absorption depth at 1450 nm: $1.0 - \frac{R_{1450}}{0.5(R_{1300} + R_{1600})}$.
- **`SHAPE_water_depth_1940`**: Continuum-relative absorption depth at 1940 nm: $1.0 - \frac{R_{1940}}{0.5(R_{1800} + R_{2100})}$.
- **`SHAPE_sr_nir_red`**: Simple Ratio $\frac{R_{800}}{R_{680}}$.
- **`SHAPE_sr_nir_green`**: Simple Ratio $\frac{R_{800}}{R_{550}}$.
- **`SHAPE_peak_reflectance`**: Global maximum reflectance value across all bands.
- **`SHAPE_peak_wavelength_nm`**: Wavelength corresponding to peak reflectance.

### D. Global & Region-Stratified Statistical Moments (22 Features)

- **Global Moments (7 features):** Mean, Standard Deviation, Median, Skewness, Kurtosis, Interquartile Range (IQR), and Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) computed across all 653 bands.
- **Regional Moments (15 features):** Mean, Standard Deviation, and IQR stratified across 5 biophysical regimes:
  1. *Visible (VIS):* $450 - 700\text{ nm}$
  2. *Red-Edge (RE):* $680 - 750\text{ nm}$
  3. *Near-Infrared (NIR):* $750 - 1300\text{ nm}$
  4. *Short-Wave Infrared 1 (SWIR1):* $1300 - 1900\text{ nm}$
  5. *Short-Wave Infrared 2 (SWIR2):* $1900 - 2400\text{ nm}$

---

## 🛡️ 3. Defensive Data Leakage Architecture

### A. The Biological Grouping Imperative
In agricultural hyperspectral experiments, each physical plant specimen (`barcode`) is repeatedly scanned over multiple days (7–8 scans per plant). A naive random observation-level train/test split causes **massive biological data leakage**: identical plants appear in both train and test splits, causing models to memorize plant-specific genotypes rather than learning general disease pathology.

### B. Stratified Group Partitioning
We implemented `StratifiedGroupSplitter`, which:
1. Identifies unique plant specimens (208 total: 104 Control, 104 Infected).
2. Partitions barcodes into **70% Train, 15% Validation, 15% Test** while stratifying by disease class.
3. Maps group assignments back to observations.

```
Total Observations : 1,632 scans across 208 physical plants
├── Train Split    : 1,144 observations (145 plants: 73 Control, 72 Infected)  [70.1%]
├── Val Split      :   244 observations ( 31 plants: 15 Control, 16 Infected)  [14.95%]
└── Test Split     :   244 observations ( 32 plants: 16 Control, 16 Infected)  [14.95%]
```

### C. Zero-Leakage Audit
The `DataLeakageAuditor` rigorously verifies:
- `Train ∩ Val = ∅` (0 plant barcodes overlap)
- `Train ∩ Test = ∅` (0 plant barcodes overlap)
- `Val ∩ Test = ∅` (0 plant barcodes overlap)
- Forbidden target/ID columns (`disease`, `barcode`, `scan_id`) are completely removed from feature matrices.
- All selectors and transformers (Variance, Correlation, Random Forest, PCA, Scaler) are fitted **strictly on the training split** and applied out-of-sample.

---

## 🎯 4. Multi-Stage Feature Selection Cascade

To eliminate redundancy while maximizing disease discriminative power, features pass through a 3-stage cascade:

```
[51 Extracted Domain Features]
        │
        ▼ (Stage 1: Variance Filter, threshold = 1e-5)
[42 Non-Constant Features]  (Dropped 9 near-constant features)
        │
        ▼ (Stage 2: Correlation Filter, |r| < 0.95)
[18 Non-Collinear Features]  (Dropped 24 highly collinear redundant features)
        │
        ▼ (Stage 3: Random Forest Gini Importance, Top-K = 35)
[18 Final Predictive Features]
```

### Triangulated Feature Rankings
The 18 final selected features were validated across three distinct statistical methodologies:

| Rank | Random Forest (Gini Importance) | Mutual Information (MI) | ANOVA F-Statistic ($F$-score) | Primary Biological Function |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`VI_PRI`** (0.1341) | **`VI_PRI`** (0.334) | **`VI_PSRI`** (741.2) | Xanthophyll cycle stress & LUE |
| **2** | **`VI_PSRI`** (0.1065) | **`VI_PSRI`** (0.312) | **`VI_mSR705`** (689.4) | Senescence & tissue necrosis |
| **3** | **`VI_mSR705`** (0.0912) | `STAT_nir_std` (0.285) | `SHAPE_sr_nir_red` (612.8) | Red-Edge chlorophyll content |
| **4** | **`SHAPE_sr_nir_red`** (0.0881) | `SHAPE_peak_wavelength_nm` (0.241) | `STAT_nir_iqr` (589.1) | Canopy biomass & cellular ratio |
| **5** | `STAT_global_kurtosis` (0.0764) | `VI_mSR705` (0.239) | `STAT_re_std` (544.7) | Global spectral tails & distortion |

---

## 📉 5. Unsupervised Dimensionality Reduction (PCA)

- **Algorithm:** `StandardScaler` + `PCA(n_components=10)` fitted strictly on training partition.
- **Top 2 Components Explained Variance:** **87.49%**
- **Top 10 Components Cumulative Variance:** **99.52%**
- **Artifact:** Model serialized to `outputs/dimensionality_reduction/fitted_pca.joblib` for instant inference in `05_Machine_Learning`.

---

## 📂 6. Directory Structure

```
04_Feature_Engineering/
│
├── configs/
│   └── feature_config.json              # Master configuration
│
├── extraction/
│   ├── __init__.py
│   ├── vegetation_indices.py            # 10 narrow-band indices
│   ├── derivative_features.py           # 1st/2nd derivatives & REP
│   ├── spectral_features.py             # Continuum water depths & AUC
│   └── statistical_features.py          # Global & regional moments
│
├── selection/
│   ├── __init__.py
│   ├── variance_filter.py               # Leakage-free variance thresholding
│   ├── correlation_filter.py            # Collinearity suppression (|r| < 0.95)
│   ├── model_importance.py             # Random Forest Gini selector
│   ├── mutual_information.py           # Non-linear MI selector
│   └── statistical_selection.py         # Univariate ANOVA F selector
│
├── dimensionality_reduction/
│   ├── __init__.py
│   └── pca.py                           # Leakage-free StandardScaler + PCA
│
├── validation/
│   ├── __init__.py
│   ├── split_generator.py               # Stratified plant barcode splitter
│   ├── leakage_check.py                 # Defensive data leakage auditor
│   └── feature_validation.py            # Quality auditor (NaNs, Infs, constants)
│
├── visualizations/
│   ├── __init__.py
│   └── feature_plots.py                 # Publication-grade visualization suite
│
├── utils/
│   ├── __init__.py
│   └── feature_utils.py                 # Loaders, logging, safe I/O, SHA-256
│
├── tests/
│   ├── conftest.py                      # Synthetic mock fixtures
│   ├── test_vegetation_indices.py       # Unit tests for VIs
│   ├── test_derivative_features.py      # Unit tests for derivatives & REP
│   ├── test_spectral_features.py        # Unit tests for shape & water depths
│   ├── test_statistical_features.py     # Unit tests for moments
│   ├── test_feature_selection.py        # Unit tests for selectors
│   ├── test_pca.py                      # Unit tests for PCA & joblib
│   ├── test_leakage.py                  # Unit tests for group isolation
│   └── test_pipeline.py                 # Full end-to-end integration test
│
├── outputs/
│   ├── extracted_features/              # 51 extracted features (CSV + Parquet)
│   ├── selected_features/               # 18 selected features (CSV + Parquet)
│   ├── dimensionality_reduction/        # 10 PCA scores + fitted_pca.joblib
│   ├── feature_importance/              # RF, MI, and ANOVA rankings
│   ├── splits/                          # Disjoint split assignments & metadata
│   ├── reports/                         # Audit reports & SHA-256 manifest
│   └── visualizations/                  # 300-DPI publication figures
│
├── feature_extraction.py                # Standalone extraction CLI
├── feature_selection.py                 # Standalone selection & PCA CLI
├── feature_engineering.py               # Master end-to-end pipeline orchestrator
└── README.md                            # Comprehensive technical documentation
```

---

## 💻 7. Execution Guide & Reproduction Commands

To run the pipeline and tests, open PowerShell in the project root:

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Run Standalone Feature Extraction CLI
python 04_Feature_Engineering/feature_extraction.py

# 3. Run Standalone Leakage-Free Feature Selection & PCA CLI
python 04_Feature_Engineering/feature_selection.py

# 4. Run Complete End-to-End Master Pipeline Orchestrator
python 04_Feature_Engineering/feature_engineering.py

# 5. Execute Complete Automated Test Suite (19 tests)
python -m pytest 04_Feature_Engineering/tests -v
```

---

## 📋 8. Authoritative Artifact Catalog & SHA-256 Hashes

| Relative File Path | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :---: | :--- | :--- |
| `outputs/extracted_features/extracted_features.parquet` | 779,524 | `737cfac6712b1a5850e5bd5ced879e35826a422488df8477ab9837dc7e04952a` | All 51 features + metadata |
| `outputs/extracted_features/engineered_features_only.parquet` | 709,101 | `bf0deb1578760540ceb07e7c119103a967e451d22a82a4b0bacc077b0bc3deaf` | 51 features matrix only |
| `outputs/selected_features/selected_features.parquet` | 318,680 | `72d3a998b727d981bef759e5f40778ff04cea21e2262a228c9f4a180ddac49e8` | 18 selected features + meta + split |
| `outputs/selected_features/selected_features_only.parquet` | 247,490 | `26473b3d3ccf4605a97e58837d4f8e847f0a2224fdf315797c09a4fcb334fc82` | 18 selected features matrix only |
| `outputs/dimensionality_reduction/fitted_pca.joblib` | 9,375 | `2df278ce62019348f0b94e897bd053fcca386b9042cb85eccc477c12860a9ad3` | Serialized scaler + PCA model |
| `outputs/dimensionality_reduction/reduced_features.parquet` | 216,813 | `c230325b3718c1c3e23039a3a30df8fc53d09ca751fda88cf3f5f45bc60368de` | 10 PCA scores + meta + split |
| `outputs/splits/split_assignments.csv` | 46,348 | `07f326c68466b8037dcf9af199de87bab8b9180885d795699a29c391bd3db3a1` | Exact sample-to-split partition map |
| `outputs/splits/split_metadata.json` | 1,320 | `318cbd9b1effe9e1dc93abdf42544574740b357a08c6c7d295903c0a4dccf3d8` | Split balance & zero-leakage audit |
| `outputs/feature_importance/feature_importance.csv` | 1,037 | `4f6cbfce5a1eca4e285b759fc7062c460005c8e888878d4c3e358df2ba3f0187` | Random Forest Gini rankings |
| `outputs/feature_importance/mutual_information_ranking.csv` | 654 | `11749561c2f5d308193e6b629248c5fe6197f9211b69f1fc4c86b713f8d782a1` | Mutual Information rankings |
| `outputs/feature_importance/statistical_f_score_ranking.csv` | 1,024 | `4f5f79534f0106052370ee934d8ad255cf4477f589715d3b22039047052a90e7` | ANOVA F-statistic rankings |
| `outputs/reports/feature_engineering_pipeline_report.json` | 5,680 | `47f3be9853ebc83785a3c9429598b953b05f6b2ffad461705e4681313437299a` | Full pipeline audit report |
| `outputs/visualizations/feature_importance_ranking.png` | 199,181 | `44ad1662e06a628ff65af75b631d2313247b1f9a01a9ef940317761d52572df2` | Top 25 feature importance plot |
| `outputs/visualizations/spectral_derivatives_profile.png` | 278,422 | `c4a4119379b19a63e7e2a72220d42b8cd285ec33c9275f75b62250c04538de42` | Control vs Infected 1st derivative |
| `outputs/visualizations/pca_scree_and_cumulative_variance.png` | 206,637 | `741eb6f3d0f793966be63d4aadcba0f54f71e4a560e563955260dfbdc2117a3c` | PCA Scree & variance curves |
| `outputs/visualizations/feature_selection_stages_funnel.png` | 125,254 | `ef346a1bec75244ddecf80621bba01060e19ae7290b0c75af4eab14e9cfacf90` | Feature retention funnel |

---

## 🚀 9. Seamless Integration with `05_Machine_Learning`

The downstream model training module (`05_Machine_Learning`) can directly consume the prepared datasets with complete confidence:
1. **Selected Features Dataset:** `outputs/selected_features/selected_features.parquet` provides the optimal 18 non-collinear, high-importance predictors with zero data leakage.
2. **PCA Transformed Dataset:** `outputs/dimensionality_reduction/reduced_features.parquet` provides 10 orthogonal components capturing 99.52% of total spectral variance.
3. **Partition Labels:** Both datasets include the `split_partition` column (`train`, `val`, `test`), guaranteeing that models are evaluated on strictly unseen physical plant specimens.
