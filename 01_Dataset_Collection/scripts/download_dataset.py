"""
01_Dataset_Collection/scripts/download_dataset.py

Downloads the primary authentic hyperspectral dataset:
- Dataset: Wheat Crown Rot Hyperspectral Dataset (Figshare DOI: 10.25909/17075138)
- Target: raw_data/hyperspectral/data_AgEng_review_paper.xlsx
- Verified Expected Size: 13,333,673 bytes
"""

import sys
from pathlib import Path
from datetime import datetime
import json

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import (
    HYPERSPECTRAL_RAW_DIR,
    METADATA_RAW_DIR,
    PRIMARY_DATASET_ID,
    PRIMARY_DATASET_URL,
    PRIMARY_DATASET_FILENAME,
    PRIMARY_DATASET_EXPECTED_SIZE,
    COLLECTION_REPORT_FILE,
    setup_logger,
)
from utils.file_utils import download_file
from utils.metadata_utils import calculate_sha256, format_bytes, save_json

logger = setup_logger("download_dataset")


def run_download(force_overwrite: bool = False) -> bool:
    """Executes the dataset download and records collection report."""
    target_path = HYPERSPECTRAL_RAW_DIR / PRIMARY_DATASET_FILENAME
    logger.info(f"Target destination: {target_path}")

    last_logged_pct = [-1]

    def progress_tracker(downloaded: int, total: int):
        pct = int((downloaded / total) * 100)
        if pct % 20 == 0 and pct != last_logged_pct[0]:
            last_logged_pct[0] = pct
            logger.info(f"Download Progress: {pct}% ({format_bytes(downloaded)} / {format_bytes(total)})")

    start_time = datetime.now()
    success = download_file(
        url=PRIMARY_DATASET_URL,
        destination=target_path,
        expected_size=PRIMARY_DATASET_EXPECTED_SIZE,
        overwrite=force_overwrite,
        progress_callback=progress_tracker,
    )

    if not success:
        logger.error(f"Download failed for {PRIMARY_DATASET_ID}.")
        return False

    # Compute checksum and verify file integrity
    sha256_hash = calculate_sha256(target_path)
    file_size = target_path.stat().st_size
    duration_sec = round((datetime.now() - start_time).total_seconds(), 2)

    logger.info(f"Download verified! Size: {format_bytes(file_size)}, SHA256: {sha256_hash[:16]}...")

    # Write collection report
    report = {
        "dataset_id": PRIMARY_DATASET_ID,
        "dataset_name": "Hyperspectral Imaging for Early Detection of Crown Rot in Wheat",
        "file_name": PRIMARY_DATASET_FILENAME,
        "local_path": str(target_path.relative_to(MODULE_DIR)),
        "source_url": PRIMARY_DATASET_URL,
        "doi": "10.25909/17075138",
        "download_timestamp": datetime.now().isoformat(),
        "download_duration_seconds": duration_sec,
        "file_size_bytes": file_size,
        "file_size_human": format_bytes(file_size),
        "sha256_checksum": sha256_hash,
        "status": "DOWNLOADED_AND_VERIFIED",
    }

    save_json(report, COLLECTION_REPORT_FILE)

    print("=" * 70)
    print(" ✅ DATASET DOWNLOAD SUCCESSFUL")
    print(f"  * File: {target_path}")
    print(f"  * Size: {format_bytes(file_size)} ({file_size} bytes)")
    print(f"  * SHA-256: {sha256_hash}")
    print(f"  * Report: {COLLECTION_REPORT_FILE}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_download(force_overwrite=force)

