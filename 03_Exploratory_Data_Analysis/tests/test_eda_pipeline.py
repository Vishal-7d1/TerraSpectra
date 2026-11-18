"""
03_Exploratory_Data_Analysis/tests/test_eda_pipeline.py

End-to-end integration test for the 03_Exploratory_Data_Analysis module.
"""

from pathlib import Path
import json
import pandas as pd
from eda import run_eda_pipeline


def test_end_to_end_eda_pipeline(synthetic_csv_file: Path, tmp_path: Path):
    """Verifies that the central EDA pipeline executes from config to reports."""
    out_dir = tmp_path / "test_eda_outputs"
    test_config = {
        "pipeline_name": "Test_EDA_Pipeline",
        "version": "1.0.0",
        "data_inputs": {
            "cleaned_data_path": str(synthetic_csv_file),
            "cleaned_data_csv": str(synthetic_csv_file),
            "normalized_data_path": str(synthetic_csv_file),
            "target_disease_column": "disease",
            "watering_column": "watering",
            "cultivar_column": "genotype ID",
            "species_column": "Plant Species",
            "sample_id_column": "barcode"
        },
        "spectral_regions": {
            "visible": [450.0, 700.0],
            "red_edge": [680.0, 750.0],
            "near_infrared": [750.0, 950.0]
        },
        "vegetation_indices": {},
        "statistical_testing": {"alpha": 0.05, "fdr_method": "fdr_bh"},
        "pca": {"n_components": 3, "random_state": 42},
        "outlier_detection": {"iqr_multiplier": 2.5, "zscore_threshold": 3.0},
        "outputs": {
            "base_output_dir": str(out_dir),
            "save_csv": True,
            "save_json": True,
            "save_parquet": True,
            "generate_plots": False
        }
    }

    test_cfg_path = tmp_path / "test_config.json"
    with open(test_cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_config, f)

    success = run_eda_pipeline(config_path=test_cfg_path)
    assert success is True

    # Check key summaries generated
    assert (out_dir / "summary" / "eda_insights.json").exists()
    assert (out_dir / "summary" / "eda_summary.md").exists()
    assert (out_dir / "summary" / "key_findings.csv").exists()
    assert (out_dir / "summary" / "eda_manifest.csv").exists()
