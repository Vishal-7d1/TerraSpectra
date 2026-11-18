"""
03_Exploratory_Data_Analysis/tests/test_statistical_analysis.py

Unit tests for statistical exploratory analysis module using synthetic fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from analysis.statistical_analysis import SpectralStatisticalAnalyzer


def test_compute_band_statistics(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    analyzer = SpectralStatisticalAnalyzer(metadata_df, spectral_matrix, wavelengths)

    band_df = analyzer.compute_band_statistics()
    assert len(band_df) == len(wavelengths)
    assert "mean" in band_df.columns
    assert "variance" in band_df.columns
    assert "iqr" in band_df.columns
    assert "skewness" in band_df.columns
    assert "coefficient_of_variation" in band_df.columns

    # Reflectance should be positive
    assert np.all(band_df["mean"] > 0)
    assert np.all(band_df["std"] >= 0)


def test_grouped_and_sample_statistics(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "stats_output"

    analyzer = SpectralStatisticalAnalyzer(metadata_df, spectral_matrix, wavelengths)
    summary = analyzer.run_statistical_analysis(config, out_dir)

    assert "global_reflectance_mean" in summary
    assert (out_dir / "band_statistics.csv").exists()
    assert (out_dir / "disease_grouped_statistics.csv").exists()
    assert (out_dir / "sample_level_statistics.csv").exists()
    assert (out_dir / "statistical_summary.json").exists()

    sample_df = pd.read_csv(out_dir / "sample_level_statistics.csv")
    assert len(sample_df) == 20
    assert np.all(sample_df["spectral_auc"] > 0)
