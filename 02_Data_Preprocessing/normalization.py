"""
02_Data_Preprocessing/normalization.py

Standalone and modular normalization pipeline for TerraSpectra:
1. Loads cleaned spectral data.
2. Applies Standard Normal Variate (SNV) to remove surface scatter.
3. Implements strict Fit/Transform separation for parametric scalers to prevent data leakage.
4. Persists fitted scaler parameters.
5. Generates comparison visualizations and normalization report.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODULE_DIR))

from utils.preprocessing_utils import (
    setup_logger,
    load_config,
    save_json,
    save_dataframe,
)
from loaders.data_loader import HyperspectralDataLoader
from processors.spectral_correction import SpectralNormalizer

logger = setup_logger("normalization")


def run_normalization(config_path: Path = None) -> bool:
    """Executes the normalization stage."""
    config = load_config(config_path)
    logger.info("=" * 65)
    logger.info("STARTING SPECTRAL NORMALIZATION PIPELINE")
    logger.info("=" * 65)

    output_base_dir = MODULE_DIR / config["output"]["base_output_dir"]
    cleaned_dir = output_base_dir / "cleaned_data"
    normalized_dir = output_base_dir / "normalized_data"
    reports_dir = output_base_dir / "preprocessing_reports"
    viz_dir = output_base_dir / "visualizations"

    # 1. Locate cleaned dataset
    cleaned_file_parquet = cleaned_dir / "cleaned_spectral_data.parquet"
    cleaned_file_csv = cleaned_dir / "cleaned_spectral_data.csv"

    if cleaned_file_parquet.exists():
        loader = HyperspectralDataLoader(cleaned_file_parquet)
    elif cleaned_file_csv.exists():
        loader = HyperspectralDataLoader(cleaned_file_csv)
    else:
        logger.warning("Cleaned dataset not found in outputs/cleaned_data. Falling back to raw dataset...")
        raw_path = (MODULE_DIR / config["input"]["raw_dataset_path"]).resolve()
        loader = HyperspectralDataLoader(raw_path, sheet_name=config["input"]["sheet_name"])

    metadata_df, spectral_matrix, wavelengths, spectral_cols = loader.load()

    # 2. Prevent Data Leakage (Fit/Transform design)
    method = config["normalization"].get("method", "snv")
    normalizer = SpectralNormalizer(method=method)

    if config["normalization"].get("fit_train_only", True):
        # Determine train indices using stratified or random split
        n_samples = len(spectral_matrix)
        train_ratio = config["normalization"].get("train_ratio", 0.8)
        n_train = int(n_samples * train_ratio)
        seed = config["normalization"].get("random_seed", 42)

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(n_samples)
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]

        logger.info(f"Fitting normalizer '{method}' strictly on training set ({len(train_indices)} samples)...")
        normalizer.fit(spectral_matrix[train_indices])
        normalized_matrix = normalizer.transform(spectral_matrix)
    else:
        logger.info(f"Fitting normalizer '{method}' on entire dataset...")
        normalized_matrix = normalizer.fit_transform(spectral_matrix)

    # 3. Save Fitted Scaler
    scaler_save_path = normalized_dir / "fitted_scaler.joblib"
    normalizer.save(scaler_save_path)

    # 4. Assemble Normalized DataFrame
    norm_spectral_df = pd.DataFrame(normalized_matrix, columns=spectral_cols, index=metadata_df.index)
    normalized_df = pd.concat([metadata_df, norm_spectral_df], axis=1)

    # 5. Save Normalized Dataset
    save_paths = save_dataframe(
        normalized_df,
        base_path_no_ext=normalized_dir / "normalized_spectral_data",
        save_csv=config["output"]["save_csv"],
        save_parquet=config["output"]["save_parquet"],
    )

    # 6. Generate Comparative Visualization
    if config["output"]["generate_visualizations"]:
        plt.figure(figsize=(11, 5))
        sample_idx = 0
        plt.subplot(1, 2, 1)
        plt.plot(wavelengths, spectral_matrix[sample_idx], color="navy", lw=1.8)
        plt.title("Sample Before Normalization (Reflectance)", fontsize=11, fontweight="bold")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.grid(True, linestyle="--", alpha=0.4)

        plt.subplot(1, 2, 2)
        plt.plot(wavelengths, normalized_matrix[sample_idx], color="darkorange", lw=1.8)
        plt.title(f"Sample After Normalization ({method.upper()})", fontsize=11, fontweight="bold")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Units")
        plt.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        viz_path = viz_dir / "03_normalization_methods_comparison.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=300)
        plt.close()
        logger.info(f"Saved normalization comparison plot to: {viz_path}")

    # 7. Normalization Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "normalization_method": method,
        "fitted_on_train_only": config["normalization"].get("fit_train_only", True),
        "total_samples": len(normalized_df),
        "total_spectral_bands": len(wavelengths),
        "fitted_scaler_path": str(scaler_save_path.relative_to(MODULE_DIR)),
        "normalized_statistics": {
            "min": round(float(np.min(normalized_matrix)), 4),
            "max": round(float(np.max(normalized_matrix)), 4),
            "mean": round(float(np.mean(normalized_matrix)), 4),
            "std": round(float(np.std(normalized_matrix)), 4),
        },
        "saved_artifacts": save_paths,
        "status": "SUCCESS",
    }
    save_json(report, reports_dir / "normalization_report.json")

    print("=" * 65)
    print(" ✅ SPECTRAL NORMALIZATION COMPLETE")
    print(f"  * Method          : {method.upper()}")
    print(f"  * Normalized Mean : {report['normalized_statistics']['mean']}")
    print(f"  * Normalized Std  : {report['normalized_statistics']['std']}")
    print(f"  * Saved Artifacts : {normalized_dir}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_normalization()
    sys.exit(0 if success else 1)
