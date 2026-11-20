"""
04_Feature_Engineering/tests/test_vegetation_indices.py

Unit tests for narrow-band physiological and vegetation index extraction.
"""

import pytest
import numpy as np
import pandas as pd
from extraction.vegetation_indices import VegetationIndexExtractor


def test_vegetation_indices_extraction_shape_and_columns(mock_spectral_matrix, mock_wavelengths):
    """Verifies that exactly 10 vegetation indices are computed with correct sample count."""
    extractor = VegetationIndexExtractor(mock_spectral_matrix, mock_wavelengths)
    df, meta = extractor.extract_indices()

    expected_indices = [
        "VI_NDVI", "VI_NDRE", "VI_PRI", "VI_mSR705", "VI_WBI",
        "VI_CRI", "VI_PSRI", "VI_GNDVI", "VI_MCARI", "VI_SIPI"
    ]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(mock_spectral_matrix)
    assert len(df.columns) == 10
    for idx_name in expected_indices:
        assert idx_name in df.columns, f"Missing index: {idx_name}"


def test_vegetation_indices_finite_and_ranges(mock_spectral_matrix, mock_wavelengths):
    """Verifies that all extracted indices are finite numbers within physical ranges."""
    extractor = VegetationIndexExtractor(mock_spectral_matrix, mock_wavelengths)
    df, _ = extractor.extract_indices()

    # Zero NaNs or Infs
    assert not df.isna().any().any(), "Extracted vegetation indices contain NaNs!"
    assert not np.isinf(df.values).any(), "Extracted vegetation indices contain Infs!"

    # Normalized indices bounded by [-1, 1]
    assert (df["VI_NDVI"] >= -1.0).all() and (df["VI_NDVI"] <= 1.0).all()
    assert (df["VI_NDRE"] >= -1.0).all() and (df["VI_NDRE"] <= 1.0).all()
    assert (df["VI_GNDVI"] >= -1.0).all() and (df["VI_GNDVI"] <= 1.0).all()
    assert (df["VI_PRI"] >= -1.0).all() and (df["VI_PRI"] <= 1.0).all()


def test_vegetation_indices_zero_division_robustness():
    """Verifies that zero reflectance or flat reflectance doesn't crash extraction."""
    wl = np.linspace(450, 2400, 100)
    flat_zero_matrix = np.zeros((3, 100))
    extractor = VegetationIndexExtractor(flat_zero_matrix, wl)
    df, _ = extractor.extract_indices()

    assert not df.isna().any().any()
    assert not np.isinf(df.values).any()
