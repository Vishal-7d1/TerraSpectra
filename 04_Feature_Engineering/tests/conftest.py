"""
04_Feature_Engineering/tests/conftest.py

Shared pytest fixtures and synthetic data generators for unit & integration testing.
Zero production dependencies: all fixtures are self-contained.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
MODULE_DIR = TEST_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


@pytest.fixture
def mock_wavelengths() -> np.ndarray:
    """Provides a synthetic array of 120 wavelengths containing key biological bands."""
    # Ensure all target wavelengths for vegetation indices and water absorption exist:
    key_points = [
        450.0, 500.0, 510.0, 531.0, 550.0, 570.0, 670.0, 680.0, 700.0, 705.0, 715.0,
        720.0, 750.0, 790.0, 800.0, 850.0, 900.0, 970.0, 1000.0, 1200.0, 1300.0,
        1400.0, 1450.0, 1500.0, 1650.0, 1800.0, 1900.0, 1940.0, 2000.0, 2200.0, 2390.0
    ]
    interp_points = np.linspace(450.0, 2390.0, 90)
    all_wl = np.unique(np.sort(np.concatenate([key_points, interp_points])))
    return all_wl


@pytest.fixture
def mock_spectral_matrix(mock_wavelengths) -> np.ndarray:
    """Generates synthetic hyperspectral reflectance matrix (N=24 samples, P bands)."""
    np.random.seed(42)
    n_samples = 24
    n_bands = len(mock_wavelengths)
    
    matrix = np.zeros((n_samples, n_bands))
    for i in range(n_samples):
        # Base healthy vegetation curve: low in visible, sharp red-edge rise, high in NIR, water absorption dips in SWIR
        base = 0.08 + 0.5 * (1 / (1 + np.exp(-(mock_wavelengths - 710) / 20)))
        # Water absorption dips at 1450 nm and 1940 nm
        dip1 = 0.25 * np.exp(-((mock_wavelengths - 1450) ** 2) / (2 * 40 ** 2))
        dip2 = 0.30 * np.exp(-((mock_wavelengths - 1940) ** 2) / (2 * 40 ** 2))
        curve = base - dip1 - dip2
        # Disease attenuation for half the samples
        if i >= n_samples // 2:
            curve *= 0.75  # Infected lower NIR and altered red-edge
        noise = np.random.normal(0, 0.01, n_bands)
        matrix[i] = np.clip(curve + noise, 0.01, 0.99)

    return matrix


@pytest.fixture
def mock_metadata_df() -> pd.DataFrame:
    """Generates synthetic metadata dataframe for 24 observations across 6 unique plant barcodes."""
    barcodes = [f"PLANT_{i:03d}" for i in range(1, 7)]
    rows = []
    # 6 plants, 4 scans per plant = 24 observations
    for idx, bc in enumerate(barcodes):
        disease = "Control" if idx < 3 else "Infected"
        for scan in range(1, 5):
            rows.append({
                "barcode": bc,
                "disease": disease,
                "disease_label_binary": 0 if disease == "Control" else 1,
                "Replicate": 1,
                "watering": "Well-Watered" if idx % 2 == 0 else "Drought",
                "scan_id": f"{bc}_scan_{scan}",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_feature_df() -> pd.DataFrame:
    """Generates a synthetic feature matrix with known collinearity and low-variance columns."""
    np.random.seed(42)
    n = 100
    f1 = np.random.normal(10, 2, n)
    f2 = f1 * 0.99 + np.random.normal(0, 0.01, n)  # Highly collinear with f1 (r > 0.99)
    f3 = np.random.uniform(0, 1, n)
    f4 = np.ones(n) * 5.0  # Constant (variance = 0)
    f5 = np.random.exponential(1.5, n)
    f6 = f3 * -0.98 + np.random.normal(0, 0.01, n) # Highly negative collinear with f3
    
    return pd.DataFrame({
        "feat_norm_1": f1,
        "feat_collinear_1": f2,
        "feat_uniform": f3,
        "feat_constant": f4,
        "feat_exp": f5,
        "feat_collinear_2": f6,
    })
