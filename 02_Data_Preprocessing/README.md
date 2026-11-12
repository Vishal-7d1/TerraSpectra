# 02. Data Preprocessing

## Purpose
This folder handles the signal processing, radiometric calibration, and spectral quality enhancement of raw hyperspectral measurements.

## Contents
- `data_cleaning.py`: Cleans raw measurements, removes anomalous dead/bad bands, handles missing values, and excludes saturated pixels.
- `normalization.py`: Implements radiometric reflectance normalization and Standard Normal Variate (SNV) to mitigate surface scattering and illumination differences.
- `preprocessing.py`: Coordinates the full preprocessing pipeline, including Savitzky-Golay smoothing and atmospheric water absorption band removal.

