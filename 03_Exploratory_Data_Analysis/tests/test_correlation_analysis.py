"""
03_Exploratory_Data_Analysis/tests/test_correlation_analysis.py

Unit tests for correlation and outlier detection modules using synthetic fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from analysis.correlation_analysis import SpectralCorrelationAnalyzer
from analysis.outlier_analysis import SpectralOutlierDetector
from analysis.vegetation_indices import VegetationIndexExtractor


def test_correlation_analysis(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "corr_output"

    vi_extractor = VegetationIndexExtractor(metadata_df, spectral_matrix, wavelengths)
    indices_df, _ = vi_extractor.compute_all_indices(config)

    analyzer = SpectralCorrelationAnalyzer(metadata_df, spectral_matrix, wavelengths)
    summary = analyzer.run_correlation_analysis(config, out_dir, indices_df=indices_df)

    assert "highly_correlated_pairs_count" in summary
    assert (out_dir / "spectral_band_correlation.csv").exists()
    assert (out_dir / "highly_correlated_wavelength_pairs.csv").exists()
    assert (out_dir / "indices_disease_correlation.csv").exists()


def test_outlier_detection(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "outlier_output"

    detector = SpectralOutlierDetector(metadata_df, spectral_matrix, wavelengths)
    summary = detector.run_outlier_analysis(config, out_dir)

    assert summary["total_samples"] == 20
    assert "outlier_count_total" in summary
    assert (out_dir / "outlier_samples.csv").exists()
    assert (out_dir / "outlier_summary.json").exists()
