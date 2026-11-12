# 04. Feature Engineering

## Purpose
This folder focuses on transforming raw contiguous spectral bands into high-value diagnostic features and reducing dimensionality to avoid the Hughes phenomenon (curse of dimensionality).

## Contents
- `feature_extraction.py`: Computes narrow-band spectral vegetation indices sensitive to crop physiology and pathology:
  - **NDVI** (Normalized Difference Vegetation Index - General vigor)
  - **NDRE** (Normalized Difference Red Edge - Early pre-symptomatic stress)
  - **PRI** (Photochemical Reflectance Index - Photosynthetic light-use efficiency)
  - **PSRI** (Plant Senescence Reflectance Index - Cellular necrosis)
  - **GNDVI** (Green NDVI - Chlorophyll concentration)
- `feature_selection.py`: Implements dimensionality reduction techniques (e.g., Principal Component Analysis - PCA) and algorithms to identify and rank the most discriminative nanometer wavelengths.

