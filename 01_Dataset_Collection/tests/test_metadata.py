"""
Tests for metadata utilities: SHA-256 calculation, formatting, and file serialization.
"""

from pathlib import Path
import hashlib
from utils.metadata_utils import (
    calculate_sha256,
    format_bytes,
    save_json,
    load_json,
    save_csv,
)


def test_calculate_sha256(tmp_path: Path):
    """Verifies that calculate_sha256 matches hashlib output."""
    test_file = tmp_path / "sha_test.txt"
    content = b"TerraSpectra Hyperspectral Research Data"
    test_file.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()
    actual_sha = calculate_sha256(test_file)
    assert actual_sha == expected_sha


def test_format_bytes():
    """Verifies human-readable byte size representations."""
    assert format_bytes(500) == "500.0 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(1073741824) == "1.0 GB"


def test_json_serialization(tmp_path: Path):
    """Verifies JSON save and load round-trip fidelity."""
    json_path = tmp_path / "metadata_test.json"
    payload = {"dataset_id": "TEST_001", "samples": 100, "verified": True}
    save_json(payload, json_path)
    loaded = load_json(json_path)
    assert loaded == payload


def test_csv_serialization(tmp_path: Path):
    """Verifies CSV save operations."""
    csv_path = tmp_path / "registry_test.csv"
    records = [
        {"id": "1", "name": "Set A"},
        {"id": "2", "name": "Set B"}
    ]
    save_csv(records, csv_path, ["id", "name"])
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,name"
    assert len(lines) == 3

