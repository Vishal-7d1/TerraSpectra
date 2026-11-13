"""
Pytest configuration and isolated fixtures for 01_Dataset_Collection tests.
NOTE: All test fixtures are strictly synthetic, temporary, and isolated in tmp_path.
No test data mixes with production dataset folders.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import zipfile

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))


@pytest.fixture
def sample_test_csv(tmp_path) -> Path:
    """Creates a small synthetic test CSV clearly labeled as test-only."""
    csv_path = tmp_path / "test_only_synthetic_spectra.csv"
    df = pd.DataFrame({
        "sample_id": ["TEST_001", "TEST_002", "TEST_003", "TEST_004"],
        "disease": ["Control", "Infected", "Control", "Infected"],
        "400.0": [0.05, 0.06, 0.05, 0.07],
        "550.0": [0.12, 0.11, 0.13, 0.10],
        "670.0": [0.04, 0.08, 0.04, 0.09],
        "720.0": [0.35, 0.28, 0.36, 0.25],
        "800.0": [0.50, 0.40, 0.52, 0.38],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_test_file(tmp_path) -> Path:
    """Creates an empty 0-byte file to test validation error handling."""
    empty_path = tmp_path / "test_only_empty_file.csv"
    empty_path.touch()
    return empty_path


@pytest.fixture
def sample_test_zip(tmp_path) -> Path:
    """Creates a safe test zip archive containing one text file."""
    zip_path = tmp_path / "test_only_archive.zip"
    inner_file = tmp_path / "test_contained_file.txt"
    inner_file.write_text("TEST ONLY FIXTURE CONTENT", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner_file, arcname="test_contained_file.txt")

    return zip_path

