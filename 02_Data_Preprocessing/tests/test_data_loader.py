"""
Tests for HyperspectralDataLoader.
"""

from pathlib import Path
import pytest
import numpy as np
from loaders.data_loader import HyperspectralDataLoader


def test_loader_loads_synthetic_csv(synthetic_spectral_dataframe: Path):
    """Verifies that HyperspectralDataLoader correctly parses metadata and spectral bands."""
    loader = HyperspectralDataLoader(synthetic_spectral_dataframe)
    metadata_df, spectral_matrix, wavelengths, spectral_cols = loader.load()

    assert len(metadata_df) == 10
    assert "barcode" in metadata_df.columns
    assert "disease" in metadata_df.columns
    assert spectral_matrix.shape == (10, 20)
    assert len(wavelengths) == 20
    assert np.all(np.diff(wavelengths) > 0)  # Monotonically increasing
    assert wavelengths[0] == pytest.approx(400.0, 0.1)
    assert wavelengths[-1] == pytest.approx(1000.0, 0.1)


def test_loader_file_not_found(tmp_path: Path):
    """Verifies proper FileNotFoundError handling."""
    loader = HyperspectralDataLoader(tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()
