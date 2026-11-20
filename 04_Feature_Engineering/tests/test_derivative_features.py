"""
04_Feature_Engineering/tests/test_derivative_features.py

Unit tests for spectral derivatives, Red-Edge Position (REP), and slope metrics.
"""

import pytest
import numpy as np
import pandas as pd
from extraction.derivative_features import SpectralDerivativeExtractor


def test_derivative_extraction_shape_and_columns(mock_spectral_matrix, mock_wavelengths):
    """Verifies that 8 derivative features are extracted with correct dimensions."""
    extractor = SpectralDerivativeExtractor(mock_spectral_matrix, mock_wavelengths)
    df, meta = extractor.extract_derivative_features()

    expected_cols = [
        "DERIV_REP_wavelength_nm", "DERIV_REP_amplitude", "DERIV_REP_area",
        "DERIV_green_slope", "DERIV_swir_water_slope_1450", "DERIV_swir_water_slope_1940",
        "DERIV_red_edge_min_curvature", "DERIV_red_edge_max_curvature"
    ]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(mock_spectral_matrix)
    assert len(df.columns) == 8
    for col in expected_cols:
        assert col in df.columns, f"Missing derivative feature: {col}"


def test_derivative_rep_bounds_and_validity(mock_spectral_matrix, mock_wavelengths):
    """Verifies that Red-Edge Position (REP) falls in the biophysical 680-750 nm zone."""
    extractor = SpectralDerivativeExtractor(mock_spectral_matrix, mock_wavelengths)
    df, _ = extractor.extract_derivative_features()

    assert not df.isna().any().any(), "Derivative features contain NaNs"
    assert not np.isinf(df.values).any(), "Derivative features contain Infs"

    rep_wl = df["DERIV_REP_wavelength_nm"].values
    assert (rep_wl >= 675.0).all() and (rep_wl <= 755.0).all(), f"REP out of physical bounds: {rep_wl}"
    assert (df["DERIV_REP_amplitude"] > 0).all(), "REP amplitude must be positive in green vegetation"


def test_derivative_matrix_shapes(mock_spectral_matrix, mock_wavelengths):
    """Verifies that internal 1st and 2nd derivative matrices have correct shape."""
    extractor = SpectralDerivativeExtractor(mock_spectral_matrix, mock_wavelengths)
    extractor.extract_derivative_features()

    assert extractor.d1_matrix.shape == mock_spectral_matrix.shape
    assert extractor.d2_matrix.shape == mock_spectral_matrix.shape
