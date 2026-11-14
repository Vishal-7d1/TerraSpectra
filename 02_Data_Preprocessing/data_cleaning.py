"""
02_Data_Preprocessing/data_cleaning.py

Standalone and modular data cleaning pipeline for TerraSpectra:
1. Validates file existence and loads raw hyperspectral measurements.
2. Checks and treats missing (NaN) and infinite (Inf) values.
3. Detects duplicate records and identifies potential spectral outliers.
4. Validates reflectance physical boundaries.
5. Standardizes target labels.
6. Saves cleaned dataset and produces cleaning report.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODULE_DIR))

from utils.preprocessing_utils import (
    setup_logger,
    load_config,
    save_json,
    save_dataframe,
    plot_spectral_comparison,
)
from loaders.data_loader import HyperspectralDataLoader
from processors.missing_value_handler import MissingValueHandler

logger = setup_logger("data_cleaning")


def run_data_cleaning(config_path: Path = None) -> bool:
    """Executes the data cleaning stage."""
    config = load_config(config_path)
    logger.info("=" * 65)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 65)

    input_rel_path = config["input"]["raw_dataset_path"]
    input_path = (MODULE_DIR / input_rel_path).resolve()
    sheet_name = config["input"]["sheet_name"]

    # 1. Load Data
    loader = HyperspectralDataLoader(data_path=input_path, sheet_name=sheet_name)
    metadata_df, spectral_matrix, wavelengths, spectral_cols = loader.load()

    # 2. Duplicate Detection
    barcode_col = config["input"].get("sample_id_column", "barcode")
    duplicate_count = 0
    if barcode_col in metadata_df.columns:
        duplicate_count = int(metadata_df.duplicated(subset=[barcode_col]).sum())
        logger.info(f"Duplicate sample IDs ({barcode_col}): {duplicate_count}")

    # 3. Missing Value & Range Cleaning
    cleaner = MissingValueHandler(
        strategy=config["cleaning"]["missing_strategy"],
        min_allowed=config["cleaning"]["min_allowed_reflectance"],
        max_allowed=config["cleaning"]["max_allowed_reflectance"],
    )
    cleaned_spectra, cleaning_audit = cleaner.clean(spectral_matrix, wavelengths)

    # 4. Outlier Detection (using spectral energy / mean reflectance IQR)
    mean_reflectance = np.mean(cleaned_spectra, axis=1)
    q25, q75 = np.percentile(mean_reflectance, [25, 75])
    iqr = q75 - q25
    multiplier = config["cleaning"].get("outlier_threshold", 2.5)
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)
    outlier_mask = (mean_reflectance < lower_bound) | (mean_reflectance > upper_bound)
    outlier_count = int(outlier_mask.sum())
    logger.info(f"Spectral outliers detected (IQR {multiplier}x threshold): {outlier_count} samples")

    # 5. Label Standardization
    label_col = config["input"].get("label_column", "disease")
    if label_col in metadata_df.columns:
        # Standardize binary label: 0 for Control/Healthy, 1 for Infected
        metadata_df["disease_binary"] = (
            metadata_df[label_col].astype(str).str.lower().str.contains("inf").astype(int)
        )
        logger.info(f"Label distribution ({label_col}): {metadata_df[label_col].value_counts().to_dict()}")

    # 6. Assemble Cleaned Dataset
    spectral_df = pd.DataFrame(cleaned_spectra, columns=spectral_cols, index=metadata_df.index)
    cleaned_df = pd.concat([metadata_df, spectral_df], axis=1)

    # 7. Save Cleaned Dataset
    output_base_dir = MODULE_DIR / config["output"]["base_output_dir"]
    cleaned_dir = output_base_dir / "cleaned_data"
    reports_dir = output_base_dir / "preprocessing_reports"
    viz_dir = output_base_dir / "visualizations"

    save_paths = save_dataframe(
        cleaned_df,
        base_path_no_ext=cleaned_dir / "cleaned_spectral_data",
        save_csv=config["output"]["save_csv"],
        save_parquet=config["output"]["save_parquet"],
    )

    # 8. Plot Visualization
    if config["output"]["generate_visualizations"]:
        plot_spectral_comparison(
            wavelengths=wavelengths,
            original_spectra=spectral_matrix,
            processed_spectra=cleaned_spectra,
            title="Raw vs. Cleaned Hyperspectral Reflectance Profiles",
            original_label="Raw Spectrum",
            processed_label="Cleaned Spectrum",
            save_path=viz_dir / "01_raw_vs_cleaned_spectra.png",
        )

    # 9. Save Audit Report
    cleaning_report = {
        "timestamp": datetime.now().isoformat(),
        "input_dataset": input_path.name,
        "total_samples": len(cleaned_df),
        "total_spectral_bands": len(wavelengths),
        "duplicate_sample_count": duplicate_count,
        "outlier_sample_count": outlier_count,
        "cleaning_audit": cleaning_audit,
        "saved_artifacts": save_paths,
        "status": "SUCCESS",
    }
    save_json(cleaning_report, reports_dir / "cleaning_report.json")

    print("=" * 65)
    print(" ✅ DATA CLEANING COMPLETE")
    print(f"  * Cleaned Samples : {len(cleaned_df):,}")
    print(f"  * Spectral Bands  : {len(wavelengths)} ({wavelengths.min():.2f} - {wavelengths.max():.2f} nm)")
    print(f"  * Duplicates      : {duplicate_count}")
    print(f"  * Outliers        : {outlier_count}")
    print(f"  * Outputs Saved   : {cleaned_dir}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_data_cleaning()
    sys.exit(0 if success else 1)
