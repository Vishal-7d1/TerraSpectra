"""
01_Dataset_Collection/scripts/validate_dataset.py

Performs comprehensive file-level and hyperspectral domain validation on downloaded datasets.
Saves validation results to dataset_info/validation_report.json.
"""

import sys
from pathlib import Path
from datetime import datetime

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import (
    HYPERSPECTRAL_RAW_DIR,
    DATASET_INFO_DIR,
    PRIMARY_DATASET_FILENAME,
    setup_logger,
)
from utils.validation_utils import validate_file_level, validate_tabular_spectral_dataset
from utils.metadata_utils import save_json

logger = setup_logger("validate_dataset")


def run_validation() -> bool:
    """Executes validation on all raw datasets in raw_data/hyperspectral/."""
    target_file = HYPERSPECTRAL_RAW_DIR / PRIMARY_DATASET_FILENAME
    logger.info(f"Initiating validation on: {target_file}")

    # 1. File-level validation
    file_val = validate_file_level(target_file)
    if file_val["status"] != "VALID":
        logger.error(f"File validation failed: {file_val['issues']}")
        return False

    # 2. Hyperspectral-specific validation
    spectral_val = validate_tabular_spectral_dataset(
        file_path=target_file,
        sheet_name="barcode_hypdata",
        label_col="disease"
    )

    combined_report = {
        "validation_timestamp": datetime.now().isoformat(),
        "target_file": str(target_file.relative_to(MODULE_DIR)),
        "file_validation": file_val,
        "hyperspectral_validation": spectral_val,
        "overall_status": "VALID" if (file_val["status"] == "VALID" and spectral_val["status"] == "VALID") else "WARNING"
    }

    report_path = DATASET_INFO_DIR / "validation_report.json"
    save_json(combined_report, report_path)

    print("=" * 70)
    print(" 🔍 HYPERSPECTRAL DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"File Name              : {spectral_val['file_name']}")
    print(f"Total Crop Samples     : {spectral_val['total_samples']:,} samples")
    print(f"Spectral Bands (Cols)  : {spectral_val['spectral_bands_count']} narrow bands")
    print(f"Wavelength Coverage    : {spectral_val['wavelength_min_nm']} nm -> {spectral_val['wavelength_max_nm']} nm")
    print(f"Reflectance Range      : [{spectral_val['min_reflectance']}, {spectral_val['max_reflectance']}]")
    print(f"NaN / Inf Counts       : {spectral_val['nan_count']} NaNs | {spectral_val['infinite_count']} Infs")
    print(f"Disease Class Breakdown: {spectral_val['classes_detected']}")
    print(f"Missing Labels         : {spectral_val['missing_labels_count']}")
    print(f"Validation Status      : {combined_report['overall_status']}")
    print(f"Saved Report           : {report_path}")
    print("=" * 70)

    return combined_report["overall_status"] in ["VALID", "WARNING"]


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)

