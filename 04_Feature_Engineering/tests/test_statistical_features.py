"""
04_Feature_Engineering/tests/test_statistical_features.py

Unit tests for global and domain-stratified statistical moment extraction.
"""

import pytest
import numpy as np
import pandas as pd
from extraction.statistical_features import StatisticalFeatureExtractor


def test_statistical_features_shape_and_columns(mock_spectral_matrix, mock_wavelengths):
    """Verifies that 22 statistical moment features are correctly extracted."""
    extractor = StatisticalFeatureExtractor(mock_spectral_matrix, mock_wavelengths)
    df, meta = extractor.extract_statistical_features()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(mock_spectral_matrix)
    assert len(df.columns) == 22
    assert "STAT_global_mean" in df.columns
    assert "STAT_global_std" in df.columns
    assert "STAT_global_skewness" in df.columns
    assert "STAT_global_kurtosis" in df.columns
    assert "STAT_global_iqr" in df.columns
    assert "STAT_vis_mean" in df.columns
    assert "STAT_re_mean" in df.columns
    assert "STAT_nir_mean" in df.columns
    assert "STAT_swir1_mean" in df.columns
    assert "STAT_swir2_mean" in df.columns


def test_statistical_features_finiteness(mock_spectral_matrix, mock_wavelengths):
    """Verifies that moments and percentiles are finite and non-empty."""
    extractor = StatisticalFeatureExtractor(mock_spectral_matrix, mock_wavelengths)
    df, _ = extractor.extract_statistical_features()

    assert not df.isna().any().any(), "Statistical features contain NaNs"
    assert not np.isinf(df.values).any(), "Statistical features contain Infs"
    assert (df["STAT_global_std"] >= 0).all()
    assert (df["STAT_nir_mean"] > 0).all()
