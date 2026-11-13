# Dataset Card: Hyperspectral Imaging for Early Detection of Crown Rot in Wheat

**Dataset ID:** `DS_WHEAT_CROWN_ROT_001`  
**Project:** TerraSpectra (Hyperspectral Crop Disease Forecasting)  
**Organization:** Infotact Solutions  
**Selection Status:** **SELECTED_PRIMARY**  
**License:** Apache 2.0 (Open Access)  
**Official Source URL:** [https://doi.org/10.25909/17075138](https://doi.org/10.25909/17075138)  
**Publication:** Xie, Y.; Plett, D.; Liu, H. The Promise of Hyperspectral Imaging for the Early Detection of Crown Rot in Wheat. AgriEngineering 2021, 3, 924-941. DOI: 10.3390/agriengineering3040057  

---

## 1. Dataset Summary
This dataset contains experimental hyperspectral reflectance measurements acquired for the early detection and diagnosis of **Crown Rot disease** in wheat (*Triticum aestivum* and *Triticum durum*), caused by the devastating fungal soil-borne pathogen ***Fusarium pseudograminearum***.

The study investigated whether hyperspectral imaging spanning the **VNIR (Visible & Near-Infrared: 400–1000 nm)** and **SWIR (Short-Wave Infrared: 1000–2400 nm)** spectrum could detect fungal invasion at the **pre-symptomatic phase** (before characteristic honey-brown stem lesions and foliar chlorosis develop).

---

## 2. Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Crop Host** | Wheat (*Triticum durum* and *Triticum aestivum*) |
| **Pathogen** | *Fusarium pseudograminearum* (Crown Rot) |
| **Total Samples** | **1,632 spectral curves** across multiple replicates |
| **Spectral Bands** | **657 contiguous narrow bands** |
| **Wavelength Range** | **396.9 nm to 2396.5 nm** (Continuous VNIR + SWIR) |
| **Spectral Resolution** | ~3.0 nm step (VNIR) to ~5.6 nm step (SWIR) |
| **Experimental Factors** | Inoculation (`Infected` vs. `Control`) & Water Regime (`Drought` vs. `Well-watered`) |
| **File Format** | Microsoft Excel (`.xlsx`), Sheet `barcode_hypdata` & Sheet `result` |
| **Original File Size** | 13,333,673 bytes (~13.3 MB) |

---

## 3. Data Integrity & Selection Justification
1. **Bio-Optical Precision:** Provides continuous narrow bands across the critical Red-Edge inflection (690–740nm) and water/lignin absorption features (1400–2100nm).
2. **Pre-Symptomatic Tracking:** Explicitly maps early fungal colonization vs. control plants under varying watering stresses.
3. **Accessibility:** Freely accessible via Figshare API/HTTP download under an Apache 2.0 open-source license.
4. **Authenticity:** Peer-reviewed and published in *AgriEngineering* (2021) by researchers at the University of Adelaide and CSIRO.
