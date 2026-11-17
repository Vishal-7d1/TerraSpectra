"""
03_Exploratory_Data_Analysis/analysis/dataset_profiling.py

Comprehensive dataset profiling for TerraSpectra hyperspectral data:
- Inspects observations, spectral bands, metadata fields, and memory footprints.
- Audits missing values, infinite values, and duplicates.
- Quantifies class balance across disease status, watering conditions, and cultivars.
- Generates structured CSV and JSON profiling reports.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from utils.eda_utils import save_json


class DatasetProfiler:
    """Profiles structure, integrity, and class distributions of hyperspectral datasets."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def run_profiling(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes complete profiling and serializes summary reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Structural Overview
        mem_mb = (self.metadata_df.memory_usage(deep=True).sum() + self.spectral_matrix.nbytes) / (1024 * 1024)
        
        overview = {
            "total_observations": int(self.num_samples),
            "total_spectral_bands": int(self.num_bands),
            "total_metadata_columns": int(self.metadata_df.shape[1]),
            "wavelength_min_nm": float(self.wavelengths.min()),
            "wavelength_max_nm": float(self.wavelengths.max()),
            "spectral_resolution_avg_nm": round(float(np.mean(np.diff(self.wavelengths))), 2),
            "estimated_memory_consumption_mb": round(float(mem_mb), 2),
        }

        # 2. Missing & Infinite Values Audit
        nan_meta = self.metadata_df.isna().sum().to_dict()
        nan_spectral = int(np.isnan(self.spectral_matrix).sum())
        inf_spectral = int(np.isinf(self.spectral_matrix).sum())

        missing_report = {
            "spectral_nan_count": nan_spectral,
            "spectral_inf_count": inf_spectral,
            "metadata_missing_counts": {k: int(v) for k, v in nan_meta.items() if v > 0},
            "is_completely_clean": (nan_spectral == 0 and inf_spectral == 0 and sum(nan_meta.values()) == 0),
        }

        # Missing values detailed CSV
        missing_df = pd.DataFrame([
            {"column": col, "missing_count": int(self.metadata_df[col].isna().sum()), "dtype": str(self.metadata_df[col].dtype)}
            for col in self.metadata_df.columns
        ])
        missing_df.to_csv(output_dir / "missing_values_report.csv", index=False)

        # 3. Duplicate Analysis
        barcode_col = config["data_inputs"].get("sample_id_column", "barcode")
        unique_barcodes = int(self.metadata_df[barcode_col].nunique()) if barcode_col in self.metadata_df.columns else self.num_samples
        duplicate_barcodes = self.num_samples - unique_barcodes

        duplicate_report = {
            "exact_duplicate_rows": int(self.metadata_df.duplicated().sum()),
            "unique_physical_plants_barcodes": unique_barcodes,
            "repeated_time_series_scans": duplicate_barcodes,
        }

        # 4. Class & Experimental Factor Distributions
        disease_col = config["data_inputs"].get("target_disease_column", "disease")
        watering_col = config["data_inputs"].get("watering_column", "watering")
        cultivar_col = config["data_inputs"].get("cultivar_column", "genotype ID")
        species_col = config["data_inputs"].get("species_column", "Plant Species")

        distributions = {}
        dist_rows = []

        for factor_name, col in [
            ("disease", disease_col),
            ("watering", watering_col),
            ("cultivar", cultivar_col),
            ("species", species_col)
        ]:
            if col in self.metadata_df.columns:
                counts = self.metadata_df[col].value_counts().to_dict()
                distributions[factor_name] = counts
                for category, count in counts.items():
                    dist_rows.append({
                        "experimental_factor": factor_name,
                        "category": str(category),
                        "count": int(count),
                        "percentage": round((count / self.num_samples) * 100, 2)
                    })

        pd.DataFrame(dist_rows).to_csv(output_dir / "class_distribution_report.csv", index=False)

        # 5. Metadata Data Types Report
        dtypes_df = pd.DataFrame([
            {"column_name": col, "data_type": str(self.metadata_df[col].dtype), "unique_values": int(self.metadata_df[col].nunique())}
            for col in self.metadata_df.columns
        ])
        dtypes_df.to_csv(output_dir / "data_types_report.csv", index=False)

        # Master Summary JSON
        full_profiling = {
            "overview": overview,
            "missing_values": missing_report,
            "duplicate_audit": duplicate_report,
            "class_distributions": distributions,
        }
        save_json(full_profiling, output_dir / "dataset_overview.json")

        return full_profiling
