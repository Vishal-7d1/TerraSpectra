"""
Metadata utilities for checksum calculation, file size formatting, and record serialization.
"""

from pathlib import Path
import hashlib
import json
import csv
from typing import Dict, Any, List

from config import setup_logger

logger = setup_logger("metadata_utils")


def calculate_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Calculates SHA-256 checksum for a file."""
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serializes data to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON artifact to: {path}")


def load_json(path: Path) -> Any:
    """Loads and deserializes JSON from file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(records: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    """Saves records to CSV with explicit fieldnames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved CSV artifact ({len(records)} rows) to: {path}")

