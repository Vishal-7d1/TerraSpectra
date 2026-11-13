"""
Tests for dataset manifest generation and catalog discovery structures.
"""

from pathlib import Path
from scripts.discover_datasets import CANDIDATE_DATASETS
from utils.metadata_utils import calculate_sha256, format_bytes


def test_candidate_datasets_catalog_structure():
    """Verifies all candidate datasets have necessary required metadata fields."""
    required_keys = [
        "dataset_id",
        "dataset_name",
        "publisher_institution",
        "official_source_url",
        "download_url",
        "crop_species",
        "disease_categories",
        "sample_count",
        "spectral_bands",
        "wavelength_range",
        "file_formats",
        "license",
        "selection_status"
    ]
    for d in CANDIDATE_DATASETS:
        for k in required_keys:
            assert k in d, f"Missing key {k} in dataset {d.get('dataset_id')}"
        assert d["sample_count"] > 0
        assert d["spectral_bands"] > 0


def test_manifest_record_structure(tmp_path: Path):
    """Verifies that individual manifest records are correctly formed."""
    mock_file = tmp_path / "hyperspectral_sample.csv"
    mock_file.write_text("sample_id,400nm,800nm\n1,0.05,0.45", encoding="utf-8")

    sha = calculate_sha256(mock_file)
    size = mock_file.stat().st_size
    size_str = format_bytes(size)

    record = {
        "file_name": mock_file.name,
        "format": mock_file.suffix,
        "size_bytes": size,
        "size_human": size_str,
        "sha256": sha
    }

    assert record["file_name"] == "hyperspectral_sample.csv"
    assert record["format"] == ".csv"
    assert record["size_bytes"] > 0
    assert len(record["sha256"]) == 64

