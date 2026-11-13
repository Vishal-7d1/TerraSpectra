"""
Tests for download pipeline behaviors, error handling, and archive security.
"""

from pathlib import Path
import pytest
from utils.file_utils import download_file, safe_extract_archive


def test_safe_archive_extraction(sample_test_zip: Path, tmp_path: Path):
    """Verifies safe archive extraction logic."""
    target_dir = tmp_path / "extracted_out"
    extracted = safe_extract_archive(sample_test_zip, target_dir)
    assert len(extracted) == 1
    assert extracted[0].exists()
    assert extracted[0].read_text(encoding="utf-8") == "TEST ONLY FIXTURE CONTENT"


def test_download_failure_handling(tmp_path: Path):
    """Verifies that an invalid/unreachable URL fails gracefully without uncaught crashes."""
    dest = tmp_path / "failed_download.dat"
    # Using non-existent domain with rapid failure
    result = download_file(
        url="http://invalid-nonexistent-domain-12345.org/file.bin",
        destination=dest,
        overwrite=True
    )
    assert result is False
    assert not dest.exists()


def test_download_duplicate_skip(tmp_path: Path):
    """Verifies that download is skipped if file already exists with matching size."""
    existing_file = tmp_path / "existing_dataset.xlsx"
    dummy_data = b"Pre-existing verified dataset content"
    existing_file.write_bytes(dummy_data)

    # Calling with overwrite=False and matching size
    result = download_file(
        url="http://any-url-will-not-be-called.org",
        destination=existing_file,
        expected_size=len(dummy_data),
        overwrite=False
    )
    assert result is True
    assert existing_file.read_bytes() == dummy_data

