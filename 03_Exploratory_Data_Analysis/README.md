# 03. Exploratory Data Analysis (EDA) Module — TerraSpectra

**TerraSpectra: Hyperspectral Crop Disease Forecasting**  
*Infotact Solutions — Data Science & Machine Learning Division*

---

## 1. Module Overview

The `03_Exploratory_Data_Analysis` (EDA) module conducts rigorous statistical, spectroscopic, and multivariate exploratory investigations on clean hyperspectral agricultural canopy data produced by `02_Data_Preprocessing`. 

Rather than generating simple cosmetic plots, this module executes evidence-based chemometric research:
1. Discovers empirical spectral reflectance signatures distinguishing healthy wheat crops from **Crown Rot (*Fusarium pseudograminearum*)** infected crops.
2. Characterizes biophysical wavelength domains (**Visible, Red-Edge, NIR, SWIR-1, and SWIR-2**).
3. Derives narrow-band physiological vegetation and water indices (**NDVI, NDRE, PRI, mSR705, WBI, CRI**).
4. Conducts whole-spectrum hypothesis testing with **Welch's t-test**, **Mann-Whitney U**, **Benjamini-Hochberg FDR correction**, and **Cohen's d effect sizes**.
5. Diagnoses inter-band collinearity and evaluates canopy dimensionality via **Unsupervised Principal Component Analysis (PCA)**.
6. Synthesizes findings into actionable, leakage-safe analytical artifacts for `04_Feature_Engineering`.

```
02_Data_Preprocessing (Cleaned Reflectance & SNV Parquet)
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            03_Exploratory_Data_Analysis Pipeline             │
├──────────────────────────────────────────────────────────────┤
│ 1. Data Ingestion & Physical Reflectance Separation          │
│ 2. Comprehensive Dataset Profiling & Balance Audits          │
│ 3. Parametric & Non-Parametric Band/Sample Statistics        │
│ 4. Biophysical Region Analysis (VIS, Red-Edge, NIR, SWIR)    │
│ 5. Healthy vs. Crown Rot Difference Spectra (ΔR, ΔR_rel)     │
│ 6. Whole-Spectrum Hypothesis Testing (Welch t-test + FDR)    │
│ 7. Narrow-Band Vegetation Indices Derivation                 │
│ 8. Collinearity & Target Feature Association (Point-Biserial)│
│ 9. Non-Destructive Outlier Screening (2.5x IQR & Z-score)    │
│ 10. Unsupervised Dimensionality Reduction (PCA Loadings)     │
│ 11. Automated Report, Artifact Manifest & Insight Synthesis  │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
              04_Feature_Engineering
```

---

## 2. Directory Architecture

```text
03_Exploratory_Data_Analysis/
│
├── configs/
│   └── eda_config.json                    # Central pipeline configuration & region parameters
│
├── analysis/
│   ├── __init__.py
│   ├── dataset_profiling.py               # Observations, bands, metadata, dtypes, & balance audits
│   ├── statistical_analysis.py            # Band-wise & sample-level parametric/non-parametric stats
│   ├── spectral_analysis.py               # Mean spectral curves & biophysical domain segmentation
│   ├── disease_analysis.py                # Difference spectra (ΔR), Welch's t-test, FDR, & Cohen's d
│   ├── vegetation_indices.py              # NDVI, NDRE, PRI, mSR705, WBI, CRI derivation & comparisons
│   ├── correlation_analysis.py            # Cross-wavelength collinearity & target correlation audits
│   ├── outlier_analysis.py                # Spectral AUC 2.5x IQR screening & anomaly profiling
│   └── pca_analysis.py                    # Unsupervised PCA, explained variance, scores & loadings
│
├── visualizations/
│   ├── __init__.py
│   ├── distribution_plots.py              # Class distributions, index boxplots, AUC histograms
│   ├── spectral_plots.py                  # Mean spectra with ±1σ/±2σ bands, region shading, CV curves
│   ├── disease_comparison_plots.py        # Healthy vs Infected curves, ΔR spectra, Manhattan FDR, Cohen's d
│   └── correlation_plots.py               # Band correlation heatmaps, index associations, PCA scree & scores
│
├── outputs/
│   ├── statistical_reports/               # Band statistics, class distributions, sample AUCs
│   ├── spectral_analysis/                 # Mean signatures, region metrics, vegetation indices parquet
│   ├── disease_analysis/                  # Difference spectra, FDR significance tests, discriminative bands
│   ├── correlation_analysis/              # Subsampled correlation matrices, collinear pairs list
│   ├── pca_analysis/                      # PCA explained variance, scores, and component loadings
│   ├── summary/                           # eda_insights.json, eda_summary.md, key_findings.csv, eda_manifest.csv
│   └── visualizations/                    # 18 publication-quality 300 DPI figures
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # Synthetic test fixtures and mock spectral arrays
│   ├── test_dataset_profiling.py          # Profiling module unit tests
│   ├── test_statistical_analysis.py       # Statistical metric unit tests
│   ├── test_spectral_analysis.py          # Spectral signature & disease testing unit tests
│   ├── test_correlation_analysis.py       # Correlation & outlier unit tests
│   ├── test_pca_analysis.py               # PCA calculation unit tests
│   └── test_eda_pipeline.py               # End-to-end integration test
│
├── utils/
│   ├── __init__.py
│   └── eda_utils.py                       # Dataset loader, wavelength mapper, Cohen's d, safe figure exporter
│
├── logs/
│   └── eda.log                            # Real-time pipeline execution logs
│
├── eda.py                                 # Main 11-step end-to-end pipeline orchestrator
├── spectral_analysis.py                   # Standalone CLI for dedicated spectral characterization
└── README.md                              # Comprehensive module documentation
```

---

## 3. Mathematical Formulations & Scientific Methodology

### 3.1 Spectral Difference & Relative Difference
To isolate wavelength-specific disease responses, mean reflectance profiles of healthy (`Control`) and infected (`Crown Rot`) canopies are contrasted:

$$\Delta R(\lambda) = \bar{R}_{\text{Infected}}(\lambda) - \bar{R}_{\text{Control}}(\lambda)$$

$$\Delta R_{\text{rel}}(\lambda) = \frac{\bar{R}_{\text{Infected}}(\lambda) - \bar{R}_{\text{Control}}(\lambda)}{\bar{R}_{\text{Control}}(\lambda)} \times 100\%$$

* **Observed Biological Pattern:** In the visible red region ($660 - 680\text{ nm}$), $\Delta R > 0$ because chlorophyll loss reduces photon absorption. In the near-infrared plateau ($750 - 1100\text{ nm}$), $\Delta R < 0$ because internal leaf mesophyll cell collapse reduces scattering.

### 3.2 Hypothesis Testing & False Discovery Rate (FDR)
Because hyperspectral data contains 653 continuous bands, calculating raw p-values across all wavelengths leads to severe Type I error inflation. To control false discoveries:
1. **Welch's Two-Sample t-test** (unequal variance) is computed for each band $\lambda_j$:
   $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
2. **Benjamini-Hochberg Procedure** corrects raw p-values into FDR adjusted $q$-values:
   $$q_{(i)} \leq \frac{i}{m} \cdot \alpha$$
3. **Standardized Effect Size (Cohen's d):**
   $$d = \frac{\bar{x}_{\text{Infected}} - \bar{x}_{\text{Control}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

### 3.3 Narrow-Band Vegetation & Water Indices
Empirical narrow-bands are selected by nearest-neighbor wavelength matching:
- **NDVI** (Normalized Difference Vegetation Index): $\frac{R_{800} - R_{680}}{R_{800} + R_{680}}$
- **NDRE** (Normalized Difference Red Edge): $\frac{R_{790} - R_{720}}{R_{790} + R_{720}}$
- **PRI** (Photochemical Reflectance Index): $\frac{R_{531} - R_{570}}{R_{531} + R_{570}}$
- **mSR705** (Modified Simple Ratio 705): $\frac{R_{750} - R_{450.8}}{R_{705} - R_{450.8}}$
- **WBI** (Water Band Index): $\frac{R_{900}}{R_{970}}$
- **CRI** (Carotenoid Reflectance Index): $\left(\frac{1}{R_{510}}\right) - \left(\frac{1}{R_{550}}\right)$

### 3.4 Unsupervised Principal Component Analysis (PCA)
To evaluate the intrinsic dimensionality of the 653 bands without label leakage:
$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}, \quad X_{\text{scaled}} = U \Sigma V^T$$
Where eigenvector loadings $V$ map principal axes back to physical nanometer bands, and sample scores $T = X_{\text{scaled}} V$ represent observations in low-dimensional orthogonal space.

---

## 4. Key Empirical Findings (Real Dataset)

| Metric / Dimension | Verified Empirical Value | Scientific Meaning |
| :--- | :--- | :--- |
| **Total Observations** | **1,632 samples** | Full sample preservation from preprocessing |
| **Spectral Bands** | **653 continuous bands** | 450.82 nm to 2396.54 nm (2.98 nm mean resolution) |
| **Disease Class Split** | **816 Infected (50%) / 816 Control (50%)** | Perfectly balanced target variable |
| **Watering Regimes** | **816 Drought / 816 Well-watered** | 50:50 orthogonal treatment design |
| **FDR Significant Bands** | **653 / 653 bands (100.0%)** | Statistically significant at $q < 0.05$ |
| **Top Effect Size Peak** | **Cohen's $d = 0.929$ (NIR / SWIR)** | Strong effect size separating infected from healthy |
| **Top Vegetation Index** | **mSR705 ($d = 1.083$) & NDRE ($d = 0.771$)** | Superior red-edge chlorophyll sensitivity over NDVI |
| **PCA Dimensionality** | **PC1: 82.35% \| PC2: 13.51% (Sum: 95.85%)** | Top 2 components retain 95.85% of total variance |
| **Spectral Outliers** | **100 samples (6.13%)** | Retained as extreme biological stress (severe necrosis/drought) |

---

## 5. Execution Guide (PowerShell / Windows)

All commands are executed from the repository root (`TerraSpectra/`):

### 5.1 Run Full 11-Step EDA Pipeline
```powershell
python 03_Exploratory_Data_Analysis/eda.py
```

### 5.2 Run Dedicated Spectral Analysis Executable
```powershell
python 03_Exploratory_Data_Analysis/spectral_analysis.py
```

### 5.3 Run Automated Pytest Test Suite
```powershell
python -m pytest 03_Exploratory_Data_Analysis/tests -v
```

---

## 6. Generated Visualizations Gallery

The pipeline automatically renders 18 publication-quality 300 DPI visualizations in `outputs/visualizations/`:

| Figure File | Focus & Analytical Purpose |
| :--- | :--- |
| `01_dataset_overview_dashboard.png` | 4-panel dashboard of disease, watering, cultivar, and species distributions |
| `02_global_spectral_signature.png` | Global canopy reflectance profile with $\pm 1\sigma$ and $\pm 2\sigma$ envelopes |
| `03_spectral_regions_breakdown.png` | Biophysical domain shading (VIS, Red-Edge, NIR, SWIR-1, SWIR-2) and water dips |
| `04_spectral_variability_cv.png` | Wavelength-wise Coefficient of Variation (%) curve across canopy samples |
| `05_healthy_vs_diseased_signatures.png` | Overlaid Control vs Crown Rot mean spectral profiles with Red-Edge highlight |
| `06_difference_spectra.png` | Dual panel showing Absolute Difference ($\Delta R$) and Relative Difference ($\Delta R_{\text{rel}} \%$) |
| `07_wavelength_significance_manhattan.png` | $-\log_{10}(q\text{-value})$ Manhattan plot across all 653 wavelengths |
| `08_cohens_d_effect_size_spectrum.png` | Standardized Cohen's d spectrum highlighting large ($|d| \geq 0.8$) effect zones |
| `09_vegetation_indices_comparison.png` | Comparative boxplots & strip plots for NDVI, NDRE, PRI, mSR705, WBI, CRI |
| `10_spectral_band_correlation_heatmap.png` | Cross-wavelength Pearson correlation heatmap showing severe collinearity clusters |
| `11_indices_disease_correlation.png` | Point-Biserial correlation horizontal bar chart ($r$ with disease infection) |
| `12_outlier_auc_distribution.png` | Integrated spectral AUC distribution with 2.5x IQR upper/lower threshold boundaries |
| `13_pca_scree_and_variance.png` | PCA Scree plot and cumulative explained variance curve (95% threshold) |
| `14_pca_scores_disease_scatter.png` | 2D projection of sample scores on PC1 vs PC2 colored by crop health |
| `15_pca_component_loadings.png` | PC1, PC2, and PC3 spectral loading weights across all 653 wavelengths |

---

## 7. Downstream Interface with `04_Feature_Engineering`

The outputs of `03_Exploratory_Data_Analysis` provide key foundational inputs for `04_Feature_Engineering`:
1. **Vegetation Indices Dataset:** `outputs/spectral_analysis/vegetation_indices_with_metadata.parquet` contains pre-computed narrow-band indices aligned with sample metadata.
2. **Collinearity Mitigation:** `outputs/correlation_analysis/highly_correlated_wavelength_pairs.csv` establishes which band clusters exhibit extreme collinearity ($|r| > 0.98$), guiding feature selection.
3. **Candidate Discriminative Bands:** `outputs/disease_analysis/wavelength_significance_tests.csv` ranks all 653 bands by Cohen's d effect size.
4. **Data Leakage Safeguard:** Supervised feature selection must strictly be performed inside cross-validation training folds during Module 04 modeling.
