"""
End-to-end pipeline integration tests for 02_Data_Preprocessing.
"""

from pathlib import Path
import json
import pytest
from preprocessing import run_pipeline


def test_end_to_end_preprocessing_on_synthetic_data(synthetic_spectral_dataframe: Path, tmp_path: Path):
    """
    Runs full 11-step preprocessing pipeline using a temporary synthetic configuration.
    Verifies that all expected outputs (cleaned data, normalized data, reports) are generated.
    """
    out_dir = tmp_path / "test_outputs"
    config = {
        "pipeline_name": "Test_Pipeline",
        "version": "1.0.0",
        "input": {
            "raw_dataset_path": str(synthetic_spectral_dataframe),
            "sheet_name": None,
            "label_column": "disease",
            "sample_id_column": "barcode"
        },
        "cleaning": {
            "enabled": True,
            "handle_missing_values": True,
            "missing_strategy": "linear_spectral_interpolation",
            "detect_duplicates": True,
            "remove_duplicates": False,
            "detect_outliers": True,
            "outlier_threshold": 2.5,
            "validate_reflectance_range": True,
            "min_allowed_reflectance": 0.0,
            "max_allowed_reflectance": 1.0
        },
        "noise_reduction": {
            "enabled": True,
            "method": "savitzky_golay",
            "sg_window_length": 5,
            "sg_polyorder": 2,
            "mask_water_absorption_bands": False
        },
        "normalization": {
            "enabled": True,
            "method": "snv",
            "fit_train_only": True,
            "train_ratio": 0.8,
            "random_seed": 42
        },
        "quality_control": {
            "enabled": True,
            "max_allowed_distortion_rmse": 0.05,
            "min_correlation_threshold": 0.98
        },
        "output": {
            "base_output_dir": str(out_dir),
            "save_csv": True,
            "save_parquet": True,
            "generate_visualizations": False
        }
    }

    test_config_path = tmp_path / "test_config.json"
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    success = run_pipeline(config_path=test_config_path)
    assert success is True

    # Verify generated output files
    assert (out_dir / "cleaned_data" / "cleaned_spectral_data.csv").exists()
    assert (out_dir / "normalized_data" / "normalized_spectral_data.csv").exists()
    assert (out_dir / "normalized_data" / "fitted_scaler.joblib").exists()
    assert (out_dir / "preprocessing_reports" / "preprocessing_summary.json").exists()
    assert (out_dir / "preprocessing_reports" / "preprocessing_report.md").exists()
