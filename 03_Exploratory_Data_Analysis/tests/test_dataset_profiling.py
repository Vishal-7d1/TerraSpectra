"""
03_Exploratory_Data_Analysis/tests/test_dataset_profiling.py

Unit tests for dataset profiling module using synthetic fixtures.
"""

from pathlib import Path
import json
import pandas as pd
from analysis.dataset_profiling import DatasetProfiler


def test_dataset_profiling(synthetic_hyperspectral_fixture):
    metadata_df, spectral_matrix, wavelengths, config, tmp_path = synthetic_hyperspectral_fixture
    out_dir = tmp_path / "profiling_output"

    profiler = DatasetProfiler(metadata_df, spectral_matrix, wavelengths)
    summary = profiler.run_profiling(config, out_dir)

    assert summary["overview"]["total_observations"] == 20
    assert summary["overview"]["total_spectral_bands"] == 30
    assert summary["missing_values"]["is_completely_clean"] is True
    assert summary["class_distributions"]["disease"]["Control"] == 10
    assert summary["class_distributions"]["disease"]["Infected"] == 10

    assert (out_dir / "dataset_overview.json").exists()
    assert (out_dir / "missing_values_report.csv").exists()
    assert (out_dir / "class_distribution_report.csv").exists()
    assert (out_dir / "data_types_report.csv").exists()
