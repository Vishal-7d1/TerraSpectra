"""
01_Dataset_Collection/scripts/generate_manifest.py

Scans raw_data/ and processed_data/ to generate a comprehensive, traceable dataset manifest.
Outputs to: dataset_info/dataset_manifest.csv
"""

import sys
from pathlib import Path
from datetime import datetime

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MANIFEST_FILE,
    setup_logger,
)
from utils.metadata_utils import calculate_sha256, format_bytes, save_csv

logger = setup_logger("generate_manifest")


def run_manifest_generation():
    """Builds detailed CSV manifest of all data files."""
    logger.info("Scanning directories to construct dataset manifest...")

    fieldnames = [
        "file_id",
        "file_name",
        "relative_path",
        "data_tier",
        "format",
        "size_bytes",
        "size_human",
        "sha256",
        "last_modified",
    ]

    records = []
    file_id = 1

    directories_to_scan = [
        (RAW_DATA_DIR, "raw_data"),
        (PROCESSED_DATA_DIR, "processed_data"),
    ]

    for base_dir, tier in directories_to_scan:
        for file_path in base_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                rel_path = file_path.relative_to(MODULE_DIR)
                stat = file_path.stat()
                sha256 = calculate_sha256(file_path)

                records.append({
                    "file_id": f"FILE_{file_id:04d}",
                    "file_name": file_path.name,
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "data_tier": tier,
                    "format": file_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "size_human": format_bytes(stat.st_size),
                    "sha256": sha256,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
                file_id += 1

    save_csv(records, MANIFEST_FILE, fieldnames)

    print("=" * 70)
    print(" 📋 DATASET MANIFEST GENERATED")
    print(f"  * Manifest Destination: {MANIFEST_FILE}")
    print(f"  * Total Files Tracked  : {len(records)}")
    for r in records:
        print(f"    - [{r['file_id']}] {r['file_name']} ({r['size_human']}) | SHA256: {r['sha256'][:12]}...")
    print("=" * 70)


if __name__ == "__main__":
    run_manifest_generation()

