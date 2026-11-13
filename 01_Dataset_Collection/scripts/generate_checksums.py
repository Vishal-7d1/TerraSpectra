"""
01_Dataset_Collection/scripts/generate_checksums.py

Generates and verifies cryptographic SHA-256 checksums for raw datasets.
Outputs to: dataset_info/checksums_sha256.json
"""

import sys
from pathlib import Path
from datetime import datetime

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import RAW_DATA_DIR, CHECKSUM_FILE, setup_logger
from utils.metadata_utils import calculate_sha256, save_json, load_json

logger = setup_logger("generate_checksums")


def generate_checksums():
    """Calculates SHA-256 for all raw data files and saves to JSON."""
    logger.info("Computing SHA-256 checksums for raw datasets...")
    checksum_map = {}

    for file_path in RAW_DATA_DIR.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            rel_path = str(file_path.relative_to(MODULE_DIR)).replace("\\", "/")
            checksum_map[rel_path] = {
                "file_name": file_path.name,
                "sha256": calculate_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
                "calculated_at": datetime.now().isoformat()
            }

    save_json(checksum_map, CHECKSUM_FILE)

    print("=" * 70)
    print(" 🔐 SHA-256 CHECKSUMS GENERATED")
    print(f"  * Checksums File: {CHECKSUM_FILE}")
    for path, data in checksum_map.items():
        print(f"    - {data['file_name']}: {data['sha256']}")
    print("=" * 70)
    return checksum_map


def verify_checksums() -> bool:
    """Verifies existing files against recorded checksums."""
    if not CHECKSUM_FILE.exists():
        logger.error(f"Checksum file not found: {CHECKSUM_FILE}. Run without --verify first.")
        return False

    records = load_json(CHECKSUM_FILE)
    all_match = True

    print("=" * 70)
    print(" 🔍 VERIFYING FILE INTEGRITY VIA SHA-256")
    print("=" * 70)

    for rel_path, meta in records.items():
        file_path = MODULE_DIR / rel_path
        if not file_path.exists():
            print(f"  ❌ MISSING: {rel_path}")
            all_match = False
            continue

        actual_sha = calculate_sha256(file_path)
        if actual_sha == meta["sha256"]:
            print(f"  ✅ MATCH: {meta['file_name']} -> {actual_sha[:16]}...")
        else:
            print(f"  ❌ MISMATCH: {meta['file_name']} (Expected {meta['sha256'][:16]}, Got {actual_sha[:16]})")
            all_match = False

    print("=" * 70)
    return all_match


if __name__ == "__main__":
    if "--verify" in sys.argv:
        success = verify_checksums()
        sys.exit(0 if success else 1)
    else:
        generate_checksums()

