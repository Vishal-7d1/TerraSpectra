"""
04_Feature_Engineering/feature_extraction.py

Standalone Feature Extraction Executable for TerraSpectra:
1. Loads cleaned physical reflectance dataset from 02_Data_Preprocessing.
2. Derives 10 narrow-band vegetation indices (NDVI, NDRE, PRI, mSR705, WBI, CRI, PSRI, GNDVI, MCARI, SIPI).
3. Computes 1st and 2nd spectral derivatives, Red-Edge position (REP), and water absorption slopes.
4. Calculates spectral shape metrics (AUC, regional slopes, continuum absorption depths).
5. Computes global and regional statistical moments across 5 biophysical domains.
6. Exports unified feature matrix to Parquet and CSV in outputs/extracted_features/.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.append(str(MODULE_DIR))

from utils.feature_utils import (
    setup_logger,
    load_config,
    load_dataset,
    save_dataframe,
    save_json,
)
from extraction.vegetation_indices import VegetationIndexExtractor
from extraction.derivative_features import SpectralDerivativeExtractor
from extraction.spectral_features import SpectralShapeExtractor
from extraction.statistical_features import StatisticalFeatureExtractor

logger = setup_logger("feature_extraction_cli")


def run_feature_extraction(config_path: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Runs the feature extraction stage and exports results."""
    config = load_config(config_path)
    logger.info("=" * 70)
    logger.info("TERRASPECTRA: HYPERSPECTRAL FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 70)

    # 1. Locate dataset
    data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_path"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_csv"]
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

    logger.info(f"Loading cleaned physical reflectance dataset: {data_path.name}...")
    metadata_df, spectral_matrix, wavelengths, _ = load_dataset(data_path)
    logger.info(f"Loaded {spectral_matrix.shape[0]} observations across {len(wavelengths)} spectral bands.")

    # 2. Extract Feature Categories
    logger.info("Extracting 10 narrow-band vegetation & physiological indices...")
    vi_extractor = VegetationIndexExtractor(spectral_matrix, wavelengths)
    vi_df, vi_meta = vi_extractor.extract_indices()

    logger.info("Computing 1st & 2nd spectral derivatives & Red-Edge Position (REP)...")
    deriv_extractor = SpectralDerivativeExtractor(spectral_matrix, wavelengths)
    deriv_df, deriv_meta = deriv_extractor.extract_derivative_features()

    logger.info("Extracting continuum absorption depths & spectral shape features...")
    shape_extractor = SpectralShapeExtractor(spectral_matrix, wavelengths)
    shape_df, shape_meta = shape_extractor.extract_shape_features()

    logger.info("Computing global & domain-stratified statistical moments...")
    stat_extractor = StatisticalFeatureExtractor(spectral_matrix, wavelengths)
    stat_df, stat_meta = stat_extractor.extract_statistical_features()

    # 3. Concatenate Feature Matrix
    engineered_df = pd.concat([vi_df, deriv_df, shape_df, stat_df], axis=1)
    logger.info(f"Extracted {engineered_df.shape[1]} engineered features across {engineered_df.shape[0]} observations.")

    # 4. Save extracted features
    out_base = MODULE_DIR / "outputs" / "extracted_features"
    out_base.mkdir(parents=True, exist_ok=True)

    # Combine with metadata for standalone analysis
    full_extracted_df = pd.concat([metadata_df.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1)

    save_dataframe(full_extracted_df, out_base / "extracted_features", save_csv=True, save_parquet=True)
    save_dataframe(engineered_df, out_base / "engineered_features_only", save_csv=True, save_parquet=True)

    metadata_summary = {
        "total_samples": int(len(engineered_df)),
        "total_extracted_features": int(engineered_df.shape[1]),
        "feature_categories": {
            "vegetation_indices": list(vi_df.columns),
            "derivative_features": list(deriv_df.columns),
            "spectral_shape_features": list(shape_df.columns),
            "statistical_features": list(stat_df.columns),
        },
        "feature_counts_by_category": {
            "vegetation_indices_count": len(vi_df.columns),
            "derivative_features_count": len(deriv_df.columns),
            "spectral_shape_count": len(shape_df.columns),
            "statistical_features_count": len(stat_df.columns),
        }
    }
    save_json(metadata_summary, out_base / "extracted_features_metadata.json")

    logger.info("=" * 70)
    logger.info(" ✅ FEATURE EXTRACTION COMPLETE")
    logger.info(f"  * Total Features Extracted : {engineered_df.shape[1]}")
    logger.info(f"  * Saved Directory          : {out_base}")
    logger.info("=" * 70)

    return metadata_df, engineered_df, metadata_summary


if __name__ == "__main__":
    run_feature_extraction()
