"""
03_Exploratory_Data_Analysis/tests/test_pca_analysis.py

Unit tests for unsupervised PCA module using synthetic fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from analysis.pca_analysis import SpectralPCAAnalyzer


def test_pca_analysis(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "pca_output"

    analyzer = SpectralPCAAnalyzer(metadata_df, spectral_matrix, wavelengths)
    summary = analyzer.run_pca_analysis(config, out_dir)

    assert summary["n_components_fitted"] == 3
    assert summary["pc1_explained_variance_pct"] > 0
    assert summary["total_explained_variance_pct"] <= 100.0

    assert (out_dir / "pca_explained_variance.csv").exists()
    assert (out_dir / "pca_loadings.csv").exists()
    assert (out_dir / "pca_scores.csv").exists()
    assert (out_dir / "pca_summary.json").exists()

    scores_df = pd.read_csv(out_dir / "pca_scores.csv")
    assert len(scores_df) == 20
    assert "PC1" in scores_df.columns
    assert "PC2" in scores_df.columns
    assert "PC3" in scores_df.columns
    assert "disease" in scores_df.columns
