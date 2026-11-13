"""
01_Dataset_Collection/scripts/extract_archives.py

Handles safe decompression of archives (.zip, .tar, .tar.gz) located in raw_data/.
Verifies paths against Zip Slip vulnerabilities.
If the downloaded dataset is already uncompressed (.xlsx, .mat, .csv), validates and logs accordingly.
"""

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import RAW_DATA_DIR, HYPERSPECTRAL_RAW_DIR, setup_logger
from utils.file_utils import safe_extract_archive

logger = setup_logger("extract_archives")


def run_extraction():
    """Scans raw_data for archives and decompresses safely."""
    logger.info(f"Scanning {RAW_DATA_DIR} for archive files...")

    archive_extensions = [".zip", ".tar.gz", ".tgz", ".tar"]
    found_archives = []

    for ext in archive_extensions:
        found_archives.extend(list(RAW_DATA_DIR.rglob(f"*{ext}")))

    if not found_archives:
        logger.info("No compressed archives (.zip, .tar.gz) found. Data is present in direct format (.xlsx / .mat / .csv).")
        print("=" * 70)
        print(" ℹ️ NO ARCHIVES TO EXTRACT")
        print("  Raw datasets are stored in native uncompressed format.")
        print(f"  Location: {HYPERSPECTRAL_RAW_DIR}")
        print("=" * 70)
        return

    for archive in found_archives:
        target_dir = archive.parent / archive.stem
        logger.info(f"Extracting {archive.name} -> {target_dir}...")
        extracted = safe_extract_archive(archive, target_dir)
        print(f"✅ Extracted {len(extracted)} files from {archive.name}")


if __name__ == "__main__":
    run_extraction()

