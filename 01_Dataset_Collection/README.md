# 01. Dataset Collection

## Purpose
This folder is dedicated to the collection, initial storage, and structural audit of hyperspectral crop disease datasets.

## Contents
- `raw_data/`: Directory reserved for the original, unmodified hyperspectral files (e.g., `.mat`, `.hdr/.img`, `.tif`, or spectral `.csv`). Kept empty until a verified real dataset is selected.
- `processed_data/`: Directory for validated, formatted, and standardized tabular or spectral matrix data ready for preprocessing.
- `dataset_analysis.py`: Script responsible for verifying dataset dimensions, metadata, band wavelength ranges, missing values, and class balance.

