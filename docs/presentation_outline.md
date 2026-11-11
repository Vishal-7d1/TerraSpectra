# Presentation (PPT) Structure: TerraSpectra

**Intern:** Vishal Yadav | **Internship Ref:** ITS/DSML/1590 | **Company:** Infotact Solutions

### Slide Deck Outline (10-12 Slides)
1. **Slide 1: Title Slide**
   - Project: TerraSpectra – Hyperspectral Crop Disease Forecasting
   - Name: Vishal Yadav (Data Associate L1. Intern)
   - Company: Infotact Solutions

2. **Slide 2: Background & Industry Need**
   - Global crop loss due to plant diseases (~20-40% annually).
   - Limitation of manual scouting and standard 3-band RGB cameras.

3. **Slide 3: What is Hyperspectral Imaging (HSI)?**
   - Comparison: RGB (3 channels) vs. Hyperspectral (Hundreds of contiguous narrow bands).
   - The concept of the "Data Cube" $(X, Y, \lambda)$.
   - Spectral signatures: How healthy vs. diseased plant tissue reflects light differently.

4. **Slide 4: System Architecture**
   - Pipeline flow: Raw Data ➡️ Calibration & Savitzky-Golay Filter ➡️ PCA & Vegetation Indices ➡️ ML Classifier ➡️ Forecasting Dashboard.

5. **Slide 5: Feature Engineering: Narrow-band Vegetation Indices**
   - NDRE (Red Edge Index) for early stress detection.
   - PRI (Photochemical Reflectance Index) for xanthophyll photosynthetic changes.
   - PSRI (Senescence Index) for cell death.

6. **Slide 6: Machine Learning Models**
   - Dimensionality reduction with PCA.
   - Comparative evaluation: Random Forest, SVM, XGBoost, and 1D-CNN.

7. **Slide 7: Experimental Results & Model Performance**
   - Accuracy, Precision, Recall, F1-Score breakdown.
   - Confusion Matrix analysis (distinguishing between healthy, early stress, and advanced necrosis).

8. **Slide 8: Feature Importance Analysis**
   - Which wavelengths (nanometers) are most sensitive to pathogen attack?
   - Significance of the 680-750nm Red-Edge shift.

9. **Slide 9: Interactive Demonstration (Streamlit)**
   - Walkthrough of the live Streamlit dashboard interface (`app.py`).

10. **Slide 10: Conclusion & Future Scope**
    - Transition from drone/field spectrometer to satellite-scale monitoring.
    - Integration into automated precision agriculture spraying drones.

11. **Slide 11: Acknowledgements & Q&A**
    - Thanks to Infotact Solutions mentors and HR team.

