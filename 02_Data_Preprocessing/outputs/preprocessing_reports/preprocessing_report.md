# Preprocessing Scientific Audit Report

**Pipeline:** TerraSpectra_Hyperspectral_Preprocessing (v1.0.0)  
**Dataset:** `data_AgEng_review_paper.xlsx` (Wheat Crown Rot Hyperspectral Dataset)  
**Execution Timestamp:** 2026-11-16T06:53:21.566847  
**Execution Time:** 16.32 seconds  
**Quality Status:** **PASSED**  

---

## 1. Dimensional & Structural Audit
- **Total Processed Samples:** **1,632** (100% preserved)
- **Spectral Bands:** **653** narrow bands (450.82 nm to 2396.54 nm)
- **Label Alignment:** **Preserved** (True)
- **Disease Class Balance:**
  * `Infected`: 816 samples
  * `Control`: 816 samples

---

## 2. Noise Reduction & Spectral Fidelity
- **Algorithm:** Savitzky-Golay (Window: 11, Poly: 2)
- **Initial Noise Proxy (sigma):** 0.000977
- **Final Noise Proxy (sigma):** 0.000208
- **Noise Reduction:** **78.71%**
- **Spectral RMSE:** `0.001088` (Distortion check: PASSED)
- **Mean Pearson Correlation with Raw:** `0.999778` (> 0.999 fidelity)

---

## 3. Normalization & Scatter Correction
- **Primary Method:** **SNV** (Standard Normal Variate)
- **Data Leakage Safeguard:** Fitted strictly on training split (Ratio: 0.8)
- **Fitted Scaler Checkpoint:** `outputs/normalized_data/fitted_scaler.joblib`

---

## 4. Generated Artifacts
- **Cleaned Data:** `outputs/cleaned_data/cleaned_spectral_data.parquet` & `.csv`
- **Normalized Data:** `outputs/normalized_data/normalized_spectral_data.parquet` & `.csv`
- **Visual Diagnostics:** `outputs/visualizations/`
- **Manifest:** `outputs/preprocessing_reports/preprocessing_manifest.csv`
