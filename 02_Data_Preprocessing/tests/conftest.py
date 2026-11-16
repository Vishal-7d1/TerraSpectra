"""
02_Data_Preprocessing/tests/conftest.py

Isolated synthetic fixtures strictly for automated unit testing.
Never mixed with real production datasets.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))


@pytest.fixture
def synthetic_spectral_dataframe(tmp_path: Path) -> Path:
    """Generates a small synthetic CSV with known spectral properties for testing."""
    csv_path = tmp_path / "test_only_synthetic_spectra.csv"
    np.random.seed(42)
    wavelengths = np.linspace(400, 1000, 20)
    samples = 10

    # Base profile: low red ~670, high NIR ~800
    spectra = np.zeros((samples, len(wavelengths)))
    for i in range(samples):
        base = 0.05 + 0.40 / (1.0 + np.exp(-(wavelengths - 720) / 25))
        noise = np.random.normal(0, 0.01, size=len(wavelengths))
        spectra[i] = np.clip(base + noise, 0.01, 0.95)

    data = {
        "barcode": [f"TEST_{i:03d}" for i in range(samples)],
        "disease": ["Control" if i % 2 == 0 else "Infected" for i in range(samples)],
        "watering": ["Well-watered" if i < 5 else "Drought" for i in range(samples)],
    }
    for j, wl in enumerate(wavelengths):
        data[f"{wl:.1f}"] = spectra[:, j]

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def noisy_spectra_with_nans():
    """Creates a raw matrix with deliberate NaNs and sensor noise for unit tests."""
    np.random.seed(101)
    wls = np.linspace(500, 900, 15)
    mat = np.tile(np.sin(wls / 100), (5, 1))
    mat[1, 3] = np.nan
    mat[3, 7] = np.nan
    mat[0, 0] = -0.05  # invalid negative reflectance
    mat[2, 10] = 1.20  # over-range reflectance
    return mat, wls
