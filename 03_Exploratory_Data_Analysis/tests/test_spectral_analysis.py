"""
03_Exploratory_Data_Analysis/tests/test_spectral_analysis.py

Unit tests for spectral profile characterization, disease differentiation,
and vegetation index derivations using synthetic fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from analysis.spectral_analysis import HyperspectralSignatureAnalyzer
from analysis.disease_analysis import DiseaseSpectralComparator
from analysis.vegetation_indices import VegetationIndexExtractor


def test_hyperspectral_signatures_and_regions(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "spectral_output"

    analyzer = HyperspectralSignatureAnalyzer(metadata_df, spectral_matrix, wavelengths)
    summary = analyzer.run_spectral_analysis(config, out_dir)

    assert summary["total_bands_analyzed"] == len(wavelengths)
    assert (out_dir / "mean_spectral_signatures.csv").exists()
    assert (out_dir / "spectral_regions_summary.csv").exists()

    reg_df = pd.read_csv(out_dir / "spectral_regions_summary.csv")
    assert "visible" in reg_df["region_name"].values
    assert "near_infrared" in reg_df["region_name"].values


def test_disease_comparison_and_significance(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "disease_output"

    comparator = DiseaseSpectralComparator(metadata_df, spectral_matrix, wavelengths)
    summary = comparator.run_disease_analysis(config, out_dir)

    assert "fdr_significant_bands_count" in summary
    assert (out_dir / "spectral_differences.csv").exists()
    assert (out_dir / "wavelength_significance_tests.csv").exists()
    assert (out_dir / "discriminative_bands_summary.json").exists()

    diff_df = pd.read_csv(out_dir / "spectral_differences.csv")
    assert len(diff_df) == len(wavelengths)
    assert "absolute_difference" in diff_df.columns
    assert "relative_difference_pct" in diff_df.columns


def test_vegetation_indices_extraction(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "indices_output"

    extractor = VegetationIndexExtractor(metadata_df, spectral_matrix, wavelengths)
    summary = extractor.run_indices_extraction(config, out_dir)

    assert "indices_calculated" in summary
    assert "NDVI" in summary["indices_calculated"]
    assert "NDRE" in summary["indices_calculated"]
    assert "PRI" in summary["indices_calculated"]

    assert (out_dir / "calculated_vegetation_indices.csv").exists()
    assert (out_dir / "indices_statistical_comparison.csv").exists()
    assert (out_dir / "vegetation_indices_mapping.json").exists()

    ind_df = pd.read_csv(out_dir / "calculated_vegetation_indices.csv")
    assert len(ind_df) == 20
    assert not ind_df.isna().any().any()
