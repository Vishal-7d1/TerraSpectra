"""
03_Exploratory_Data_Analysis/tests/conftest.py

Reusable isolated fixtures for unit testing EDA modules without using real project data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def synthetic_hyperspectral_fixture(tmp_path: Path):
    """
    Creates a synthetic hyperspectral dataset with 20 samples and 30 narrow bands.
    Includes realistic metadata and distinct healthy vs infected signatures.
    """
    np.random.seed(42)
    n_samples = 20
    wavelengths = np.linspace(450.0, 950.0, 30)

    # 10 Control, 10 Infected
    diseases = ["Control"] * 10 + ["Infected"] * 10
    waterings = ["Well-watered" if i % 2 == 0 else "Drought" for i in range(n_samples)]
    genotypes = [f"G0{i % 4 + 1}" for i in range(n_samples)]
    species = ["Triticum aestivum" if i < 15 else "Triticum durum" for i in range(n_samples)]

    metadata_df = pd.DataFrame({
        "barcode": [f"BC_{1000+i}" for i in range(n_samples)],
        "disease": diseases,
        "watering": waterings,
        "genotype ID": genotypes,
        "Plant Species": species,
    })

    # Synthetic reflectance curves
    # Control has high NIR plateau (750-950 nm) ~ 0.40; Infected has reduced NIR ~ 0.28
    spectral_matrix = np.zeros((n_samples, len(wavelengths)))
    for i in range(n_samples):
        base_curve = 0.05 + 0.35 / (1.0 + np.exp(-(wavelengths - 700.0) / 20.0))
        if diseases[i] == "Infected":
            base_curve *= 0.75  # 25% lower NIR reflectance
        noise = np.random.normal(0, 0.005, len(wavelengths))
        spectral_matrix[i] = np.clip(base_curve + noise, 0.01, 0.95)

    config = {
        "pipeline_name": "Test_EDA_Pipeline",
        "version": "1.0.0",
        "data_inputs": {
            "target_disease_column": "disease",
            "watering_column": "watering",
            "cultivar_column": "genotype ID",
            "species_column": "Plant Species",
            "sample_id_column": "barcode"
        },
        "spectral_regions": {
            "visible": [450.0, 700.0],
            "red_edge": [680.0, 750.0],
            "near_infrared": [750.0, 950.0]
        },
        "vegetation_indices": {},
        "statistical_testing": {"alpha": 0.05},
        "pca": {"n_components": 3, "random_state": 42},
        "outlier_detection": {"iqr_multiplier": 2.5, "zscore_threshold": 3.0},
        "outputs": {
            "base_output_dir": str(tmp_path / "outputs"),
            "generate_plots": False
        }
    }

    return metadata_df, spectral_matrix, wavelengths, config, tmp_path


@pytest.fixture
def synthetic_csv_file(tmp_path: Path, synthetic_hyperspectral_fixture):
    """Saves synthetic dataset to CSV and returns its path."""
    metadata_df, spectral_matrix, wavelengths, _, _ = synthetic_hyperspectral_fixture
    full_df = metadata_df.copy()
    for idx, wl in enumerate(wavelengths):
        full_df[f"{wl:.2f}"] = spectral_matrix[:, idx]

    csv_path = tmp_path / "synthetic_spectra.csv"
    full_df.to_csv(csv_path, index=False)
    return csv_path
