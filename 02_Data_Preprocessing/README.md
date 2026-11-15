# 02. Data Preprocessing Module — TerraSpectra

**TerraSpectra: Hyperspectral Crop Disease Forecasting**  
*Infotact Solutions — Data Science & Machine Learning Division*

---

## 1. Module Overview

The `02_Data_Preprocessing` module serves as the core signal-processing and spectral quality enhancement engine of the **TerraSpectra** pipeline. Raw hyperspectral imaging and spectroradiometer measurements acquired from agricultural canopies often contain sensor sensor-readout noise, atmospheric scattering, illumination variations, particle-size physical scattering, dead/saturated bands, and strong atmospheric water absorption bands.

This module ingests raw hyperspectral data from `01_Dataset_Collection`, applies industry-standard chemometric and spectroscopic algorithms, and outputs mathematically conditioned, clean, and normalized spectral tensors ready for feature engineering and machine learning.

```
01_Dataset_Collection (Raw .xlsx / .parquet)
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│               02_Data_Preprocessing Pipeline                │
├─────────────────────────────────────────────────────────────┤
│ 1. Data Ingestion & Band Alignment                          │
│ 2. Missing Value Audit & Spectral Wavelength Interpolation  │
│ 3. Physical Boundary Enforcement [0.0, 1.0]                 │
│ 4. Dead / Noisy Band Auditing                               │
│ 5. Signal Noise Reduction (Savitzky-Golay / Median / SMA)   │
│ 6. Atmospheric Water Absorption Masking (1350–1420, 1800–1950)
│ 7. Scatter Correction: Standard Normal Variate (SNV)        │
│ 8. Strict Train-Fit Normalization (StandardScaler / MinMax) │
│ 9. Comprehensive Quality Control & Metric Verification      │
│ 10. Multi-Format Serialization (Parquet + CSV + Scaler Joblib)
│ 11. Automated Report & Visualizations Generation            │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
03_Exploratory_Data_Analysis / 04_Feature_Engineering / 05_Machine_Learning
```

---

## 2. Hyperspectral Preprocessing Theory & Mathematical Formulations

### 2.1 Missing Value Handling: Linear Spectral Interpolation
For continuous narrow-band reflectance signatures, an isolated missing or corrupted band at wavelength $\lambda_i$ is interpolated from adjacent valid spectral neighbors:

$$\hat{R}(\lambda_i) = R(\lambda_{i-1}) + \frac{\lambda_i - \lambda_{i-1}}{\lambda_{i+1} - \lambda_{i-1}} \cdot \left[ R(\lambda_{i+1}) - R(\lambda_{i-1}) \right]$$

### 2.2 Noise Reduction: Savitzky-Golay Digital Filter
The Savitzky-Golay filter smoothes high-frequency sensor noise while preserving critical spectral absorption band peaks, widths, and inflection points. It fits a local polynomial of degree $d$ over a moving symmetric window of $2m + 1$ wavelength points:

$$p(\tau) = \sum_{k=0}^{d} a_k \tau^k, \quad \tau \in \{-m, \dots, 0, \dots, m\}$$

The smoothed reflectance at central point $j$ is obtained by convolving with precomputed convolution coefficients $C_k$:

$$R_{\text{smooth}}(j) = \frac{\sum_{k=-m}^{m} C_k \cdot R(j+k)}{\sum_{k=-m}^{m} C_k}$$

In our pipeline:
- Default `window_length = 11` (or 15)
- Default `polyorder = 2` (or 3)
- Optional `deriv = 0` (smoothing), `deriv = 1` (first derivative / red-edge inflection analysis).

### 2.3 Scatter Correction: Standard Normal Variate (SNV)
Physical leaf surface texture, cuticle waxes, and non-uniform illumination cause baseline shifts and multiplicative light-scattering variations across samples. SNV centers and scales each spectrum individually across its own wavelength axis:

$$\text{SNV}(R_i(\lambda_j)) = \frac{R_i(\lambda_j) - \mu_i}{\sigma_i}$$

Where for sample $i$:
$$\mu_i = \frac{1}{P} \sum_{j=1}^{P} R_i(\lambda_j), \quad \sigma_i = \sqrt{\frac{1}{P - 1} \sum_{j=1}^{P} (R_i(\lambda_j) - \mu_i)^2}$$

*(Note: SNV is calculated sample-wise across bands; it introduces zero cross-sample data leakage and can be computed independently per observation).*

### 2.4 Scaler Leakage Prevention (MinMax & StandardScaler)
When applying population-level normalization across spectral bands:
- **StandardScaler**: $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$
- **MinMaxScaler**: $x' = \frac{x - x_{\min,\text{train}}}{x_{\max,\text{train}} - x_{\min,\text{train}}}$

The scaler is strictly **fit on the training partition** and then used to **transform** both validation and test sets. The fitted scaler object is serialized to `outputs/normalized_data/fitted_scaler.joblib` for inference serving.

### 2.5 Atmospheric Water Absorption Masking
Atmospheric and dense liquid leaf moisture exhibits strong photon absorption bands centered around:
- **Water Absorption Band 1:** $1350\text{ nm} - 1420\text{ nm}$
- **Water Absorption Band 2:** $1800\text{ nm} - 1950\text{ nm}$

In field conditions, these bands often exhibit near-zero signal-to-noise ratios. The pipeline provides automated detection and masking for downstream modeling stability.

---

## 3. Directory Structure

```
02_Data_Preprocessing/
│
├── configs/
│   └── preprocessing_config.json        # Central pipeline parameters
│
├── loaders/
│   ├── __init__.py
│   └── data_loader.py                   # Hyperspectral loader & metadata extractor
│
├── processors/
│   ├── __init__.py
│   ├── missing_value_handler.py         # Audit, spectral interpolation, boundary clipping
│   ├── noise_reduction.py               # Savitzky-Golay, median, moving average filters
│   ├── spectral_correction.py           # SNV, MinMaxScaler, StandardScaler (leakage-free)
│   └── quality_control.py               # Preprocessing validation & metric audits
│
├── utils/
│   ├── __init__.py
│   └── preprocessing_utils.py           # Logging, config parser, export & plotting utils
│
├── outputs/
│   ├── cleaned_data/
│   │   ├── cleaned_spectral_data.parquet
│   │   └── cleaned_spectral_data.csv
│   ├── normalized_data/
│   │   ├── normalized_spectral_data.parquet
│   │   ├── normalized_spectral_data.csv
│   │   └── fitted_scaler.joblib
│   ├── preprocessing_reports/
│   │   ├── preprocessing_summary.json
│   │   ├── preprocessing_report.md
│   │   └── preprocessing_manifest.csv
│   └── visualizations/
│       ├── 01_raw_vs_cleaned_spectra.png
│       ├── 02_noise_reduction_comparison.png
│       ├── 03_normalization_methods_comparison.png
│       ├── 04_water_absorption_masking.png
│       └── 05_preprocessing_summary_dashboard.png
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Reusable fixtures & mock spectral cubes
│   ├── test_data_loader.py              # Ingestion unit tests
│   ├── test_data_cleaning.py            # Cleaning & interpolation unit tests
│   ├── test_normalization.py            # SNV & scaler leakage tests
│   ├── test_quality_control.py          # QC validation & metric tests
│   └── test_preprocessing.py            # End-to-end integration tests
│
├── data_cleaning.py                     # Standalone CLI for cleaning stage
├── normalization.py                     # Standalone CLI for normalization stage
├── preprocessing.py                     # Main 11-step pipeline orchestrator
└── README.md                            # Comprehensive module documentation
```

---

## 4. Pipeline Configuration (`preprocessing_config.json`)

The pipeline behavior is controlled via `configs/preprocessing_config.json`:

```json
{
  "pipeline": {
    "random_seed": 42,
    "input_raw_path": "01_Dataset_Collection/raw_data/hyperspectral/data_AgEng_review_paper.xlsx",
    "output_dir": "02_Data_Preprocessing/outputs",
    "save_parquet": true,
    "save_csv": true,
    "generate_plots": true
  },
  "data_cleaning": {
    "missing_value_strategy": "interpolate",
    "reflectance_bounds": [0.0, 1.0],
    "dead_band_std_threshold": 1e-6
  },
  "noise_reduction": {
    "enabled": true,
    "method": "savitzky_golay",
    "window_length": 11,
    "polyorder": 2,
    "deriv": 0
  },
  "water_absorption_masking": {
    "enabled": false,
    "water_bands": [
      [1350.0, 1420.0],
      [1800.0, 1950.0]
    ]
  },
  "normalization": {
    "primary_method": "snv",
    "secondary_scaler": "standard",
    "prevent_data_leakage": true,
    "train_split_ratio": 0.8
  },
  "quality_control": {
    "max_acceptable_rmse": 0.05,
    "min_acceptable_pearson_r": 0.95,
    "check_label_alignment": true
  }
}
```

---

## 5. Execution Guide (PowerShell / Windows)

All commands should be executed from the repository root (`TerraSpectra/`):

### 5.1 Execute Full 11-Step Preprocessing Pipeline
```powershell
python 02_Data_Preprocessing/preprocessing.py
```

### 5.2 Execute Standalone Cleaning Module
```powershell
python 02_Data_Preprocessing/data_cleaning.py
```

### 5.3 Execute Standalone Normalization Module
```powershell
python 02_Data_Preprocessing/normalization.py
```

### 5.4 Run Automated Unit & Integration Tests
```powershell
python -m pytest 02_Data_Preprocessing/tests -v
```

---

## 6. Output Artifacts & Manifest

Upon successful pipeline completion, the following artifacts are generated:

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `outputs/cleaned_data/cleaned_spectral_data.parquet` | Apache Parquet | High-performance columnar cleaned spectral matrix + metadata |
| `outputs/cleaned_data/cleaned_spectral_data.csv` | CSV | Standard comma-separated cleaned spectral dataset |
| `outputs/normalized_data/normalized_spectral_data.parquet` | Apache Parquet | SNV/Standard scaled normalized spectral matrix |
| `outputs/normalized_data/fitted_scaler.joblib` | Scikit-Learn Joblib | Serialized scaler parameters fit on training data |
| `outputs/preprocessing_reports/preprocessing_summary.json` | JSON | Statistical audit and execution summary |
| `outputs/preprocessing_reports/preprocessing_report.md` | Markdown | Comprehensive engineering review report |
| `outputs/preprocessing_reports/preprocessing_manifest.csv` | CSV | File sizes, row counts, and cryptographic SHA-256 hashes |
| `outputs/visualizations/*.png` | PNG (300 DPI) | Publication-quality comparison spectra and dashboards |

---

## 7. Downstream Interface: Compatibility with `03_Exploratory_Data_Analysis`

The outputs of `02_Data_Preprocessing` directly plug into `03_Exploratory_Data_Analysis`:
- **Spectral Signatures:** Cleaned continuous bands enable computing mean signature per class (`Control` vs `Infected`).
- **Vegetation Indices:** Intact narrow bands in 450–900 nm allow accurate derivation of NDVI, NDRE, PRI, and mSR705.
- **Dimensionality Reduction:** PCA and t-SNE can ingest `outputs/normalized_data/normalized_spectral_data.parquet` directly.
- **Zero Leakage:** Machine learning splits can safely use `fitted_scaler.joblib` during cross-validation.
