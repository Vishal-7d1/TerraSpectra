# Project Report Draft: TerraSpectra

**Organization:** Infotact Solutions  
**Project Name:** TerraSpectra: Hyperspectral Crop Disease Forecasting  
**Domain:** Data Science and Machine Learning  
**Intern Name:** Vishal Yadav (vyadav276203@gmail.com)  
**Role:** Data Associate L1. Intern  
**Internship Ref:** ITS/DSML/1590  
**Timeline:** 10 November 2026 – 10 December 2026 (1 Month Project Completion)

---

## 1. Executive Summary
Agricultural productivity faces substantial threats from crop pathogens, rusts, blights, and fungal infections. Traditional visual inspection and standard RGB digital photography only identify diseases once visual foliar lesions appear—by which point systemic crop damage is already underway. **TerraSpectra** utilizes **Hyperspectral Imaging (HSI)** and Machine Learning to detect subtle biochemical, photosynthetic, and cellular modifications in vegetation before visual symptoms manifest.

## 2. Problem Statement
- Visual scouting is manual, labor-intensive, and prone to human diagnostic error.
- Standard RGB imaging (3 broad bands: Red, Green, Blue) lacks the spectral resolution needed to detect internal cellular breakdown or subtle chlorophyll degradation.
- Early stage (pre-symptomatic) detection is required to prevent widespread crop loss and reduce indiscriminate pesticide use.

## 3. Technical Methodology
1. **Hyperspectral Data Ingestion:**
   - 3D Hyperspectral Data Cubes $(X \times Y \times \lambda)$ covering the Visible and Near-Infrared (VNIR, 400nm – 1000nm) and Short-Wave Infrared (SWIR, 1000nm – 2500nm).
2. **Preprocessing & Spectral Calibration:**
   - Radiometric calibration (Dark and White reference normalization).
   - Noise filtering using Savitzky-Golay smoothing.
   - Removal of noisy atmospheric water vapor absorption bands (~1350–1450nm and ~1800–1950nm).
3. **Feature Engineering & Dimensionality Reduction:**
   - Principal Component Analysis (PCA) to overcome the "Curse of Dimensionality" (Hughes Phenomenon).
   - Narrow-band Spectral Vegetation Indices (NDVI, NDRE, PRI, GNDVI, PSRI).
   - Red-Edge Position (REP) derivative dynamics.
4. **Machine Learning & Deep Learning Modeling:**
   - Baseline Classifiers: Support Vector Machines (RBF Kernel), Random Forest, and XGBoost.
   - 1D-CNN / 3D-CNN for joint spectral-spatial feature extraction.
5. **Evaluation Metrics:**
   - Stratified K-Fold Cross Validation.
   - Accuracy, Precision, Recall, Weighted F1-Score, and Confusion Matrix analysis.

## 4. Key Milestones & Timeline
- **Week 1 (10 Nov - 17 Nov):** Requirement analysis, dataset collection/synthesis, literature review.
- **Week 2 (18 Nov - 24 Nov):** Exploratory Data Analysis (EDA), noise filtering, spectral index extraction.
- **Week 3 (25 Nov - 01 Dec):** Model engineering, hyperparameter tuning, validation.
- **Week 4 (02 Dec - 10 Dec):** Streamlit demonstration dashboard, final report, PPT preparation, and review submission.

