"""
Tests for file-level and tabular spectral validation utilities.
"""

from pathlib import Path
from utils.validation_utils import validate_file_level, validate_tabular_spectral_dataset


def test_validate_file_level_success(sample_test_csv: Path):
    """Verifies that a valid existing non-empty file passes file-level validation."""
    result = validate_file_level(sample_test_csv)
    assert result["exists"] is True
    assert result["is_file"] is True
    assert result["size_bytes"] > 0
    assert result["status"] == "VALID"
    assert len(result["issues"]) == 0


def test_validate_file_level_missing(tmp_path: Path):
    """Verifies that a nonexistent file returns FAILED status."""
    nonexistent = tmp_path / "nonexistent_file.csv"
    result = validate_file_level(nonexistent)
    assert result["exists"] is False
    assert result["status"] == "FAILED"
    assert "does not exist" in result["issues"][0]


def test_validate_file_level_empty(empty_test_file: Path):
    """Verifies that an empty (0 byte) file returns FAILED status."""
    result = validate_file_level(empty_test_file)
    assert result["exists"] is True
    assert result["size_bytes"] == 0
    assert result["status"] == "FAILED"
    assert "empty" in result["issues"][0]


def test_validate_tabular_spectral_dataset(sample_test_csv: Path):
    """Verifies spectral-specific checks (bands, wavelengths, class distribution)."""
    report = validate_tabular_spectral_dataset(sample_test_csv, label_col="disease")
    assert report["status"] == "VALID"
    assert report["total_samples"] == 4
    assert report["spectral_bands_count"] == 5
    assert report["wavelength_min_nm"] == 400.0
    assert report["wavelength_max_nm"] == 800.0
    assert report["nan_count"] == 0
    assert report["infinite_count"] == 0
    assert report["classes_detected"] == {"Control": 2, "Infected": 2}
    assert report["missing_labels_count"] == 0

