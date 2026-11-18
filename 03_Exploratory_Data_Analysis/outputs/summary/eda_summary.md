# TerraSpectra: Exploratory Data Analysis (EDA) Scientific Report

**Project:** TerraSpectra – Hyperspectral Crop Disease Forecasting  
**Organization:** Infotact Solutions  
**Execution Timestamp:** 2026-11-17T20:13:11.995752  
**Execution Duration:** 20.95 seconds  
**Status:** **COMPLETED_SUCCESSFULLY**  

---

## 1. Executive Summary & Dataset Profile
- **Total Hyperspectral Samples:** **1,632**
- **Continuous Spectral Bands:** **653** narrow bands (450.82 nm to 2396.54 nm)
- **Target Disease:** Crown Rot (*Fusarium pseudograminearum*) in Wheat (*T. aestivum*, *T. durum*)
- **Disease Class Balance:** **Infected: 816** | **Control: 816**
- **Watering Treatments:** **Drought: 816** | **Well-watered: 816**
- **Data Cleanliness:** 0 missing values, 0 infinite values in preprocessed data.

---

## 2. Hyperspectral Findings & Disease Differentiation
- **Red-Edge Transition Sensitivity:** Crown Rot infected samples display significant dampening of the Red-Edge slope (680 nm - 750 nm), reflecting stress-induced chlorophyll degradation.
- **Statistical Significance:** **653 out of 653 bands (100.0%)** exhibited statistically significant differences under Welch's t-test with Benjamini-Hochberg False Discovery Rate correction ($q < 0.05$).
- **Maximum Effect Size:** Cohen's $d = 0.929$ observed in the Red-Edge inflection and SWIR water-sensitive regions.

---

## 3. Vegetation & Water Indices Evaluation
- Derived narrow-band indices: `NDVI`, `NDRE`, `PRI`, `mSR705`, `WBI`, `CRI`.
- **Top Discriminative Index:** **mSR705** achieved the highest standardized effect size ($|d| = 1.083$).
- **Chlorophyll Sensitivity:** NDRE and mSR705 demonstrated superior sensitivity to early Crown Rot stress compared to broadband NDVI due to avoiding saturation in dense canopies.

---

## 4. Collinearity & Dimensionality Reduction (PCA)
- **High Collinearity:** Severe inter-band correlation ($|r| > 0.98$) among adjacent wavelengths.
- **PCA Dimensionality:**
  - **PC1:** Explains **83.21%** of variance (canopy broadband albedo / light scattering).
  - **PC2:** Explains **12.64%** of variance (Red-Edge chlorophyll absorption & water status).
  - **Top 2 Components Cumulative Variance:** **95.85%**.

---

## 5. Downstream Integration with `04_Feature_Engineering`
- **Recommended Feature Sets:**
  1. Selected narrow-band vegetation indices (`NDRE`, `mSR705`, `NDVI`, `WBI`).
  2. Red-Edge first-derivative inflection parameters.
  3. Top 3 Principal Components scores.
  4. Non-collinear spectral band subsets selected via regularized cross-validation.
