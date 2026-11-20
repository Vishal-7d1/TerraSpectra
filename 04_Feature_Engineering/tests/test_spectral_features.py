"""
04_Feature_Engineering/tests/test_spectral_features.py

Unit tests for spectral shape features, continuum water absorption depths, and AUC.
"""

import pytest
import numpy as np
import pandas as pd
from extraction.spectral_features import SpectralShapeExtractor


def test_spectral_shape_extraction_shape_and_columns(mock_spectral_matrix, mock_wavelengths):
    """Verifies that 11 spectral shape features are correctly extracted."""
    extractor = SpectralShapeExtractor(mock_spectral_matrix, mock_wavelengths)
    df, meta = extractor.extract_shape_features()

    expected_cols = [
        "SHAPE_total_spectral_auc", "SHAPE_vis_slope", "SHAPE_red_edge_slope",
        "SHAPE_nir_plateau_mean", "SHAPE_nir_plateau_slope",
        "SHAPE_water_depth_1450", "SHAPE_water_depth_1940",
        "SHAPE_sr_nir_red", "SHAPE_sr_nir_green",
        "SHAPE_peak_reflectance", "SHAPE_peak_wavelength_nm"
    ]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(mock_spectral_matrix)
    assert len(df.columns) == 11
    for col in expected_cols:
        assert col in df.columns, f"Missing spectral shape feature: {col}"


def test_spectral_shape_values_and_finiteness(mock_spectral_matrix, mock_wavelengths):
    """Verifies that AUC, ratios, and continuum depths are finite and physically plausible."""
    extractor = SpectralShapeExtractor(mock_spectral_matrix, mock_wavelengths)
    df, _ = extractor.extract_shape_features()

    assert not df.isna().any().any(), "Spectral shape features contain NaNs"
    assert not np.isinf(df.values).any(), "Spectral shape features contain Infs"

    # AUC should be strictly positive
    assert (df["SHAPE_total_spectral_auc"] > 0).all()
    assert (df["SHAPE_nir_plateau_mean"] > 0).all()

    # Ratios should be positive
    assert (df["SHAPE_sr_nir_red"] > 0).all()
    assert (df["SHAPE_sr_nir_green"] > 0).all()
