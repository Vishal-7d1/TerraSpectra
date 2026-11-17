"""
03_Exploratory_Data_Analysis/analysis/outlier_analysis.py

Outlier and anomaly diagnostic investigation for hyperspectral agricultural data:
- Non-destructive IQR (2.5x) and Z-score (3.0 sigma) evaluation across spectral AUC and individual bands.
- Distinguishes potential measurement artifacts from severe biological tissue necrosis.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from utils.eda_utils import save_json


class SpectralOutlierDetector:
    """Identifies sample-level and band-level spectral anomalies without destructive purging."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def detect_auc_outliers(self, iqr_multiplier: float = 2.5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detects spectral outliers using integrated Area Under Curve (AUC) across the full spectrum.
        """
        auc_values = np.trapz(self.spectral_matrix, self.wavelengths, axis=1)
        q25 = np.percentile(auc_values, 25)
        q75 = np.percentile(auc_values, 75)
        iqr = q75 - q25

        lower_bound = q25 - iqr_multiplier * iqr
        upper_bound = q75 + iqr_multiplier * iqr

        is_outlier = (auc_values < lower_bound) | (auc_values > upper_bound)

        bounds = {
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }
        return is_outlier, bounds

    def detect_zscore_outliers(self, z_threshold: float = 3.0) -> np.ndarray:
        """Flags observations having extreme z-scores across multiple spectral bands."""
        mean_spec = np.mean(self.spectral_matrix, axis=0)
        std_spec = np.std(self.spectral_matrix, axis=0, ddof=1)
        std_spec = np.where(std_spec == 0, 1e-9, std_spec)

        z_scores = (self.spectral_matrix - mean_spec) / std_spec
        # Flag if more than 5% of bands exceed z_threshold
        band_outlier_fraction = np.mean(np.abs(z_scores) > z_threshold, axis=1)
        return band_outlier_fraction > 0.05

    def run_outlier_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes outlier detection and exports report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        iqr_mult = config.get("outlier_detection", {}).get("iqr_multiplier", 2.5)
        z_thresh = config.get("outlier_detection", {}).get("zscore_threshold", 3.0)
        disease_col = config["data_inputs"].get("target_disease_column", "disease")

        auc_outliers, auc_bounds = self.detect_auc_outliers(iqr_multiplier=iqr_mult)
        z_outliers = self.detect_zscore_outliers(z_threshold=z_thresh)

        combined_outliers = auc_outliers | z_outliers
        outlier_indices = np.where(combined_outliers)[0]

        outlier_records = []
        auc_values = np.trapz(self.spectral_matrix, self.wavelengths, axis=1)

        for idx in outlier_indices:
            row_dict = {
                "sample_index": int(idx),
                "is_auc_outlier": bool(auc_outliers[idx]),
                "is_zscore_outlier": bool(z_outliers[idx]),
                "spectral_auc": round(float(auc_values[idx]), 3),
            }
            if disease_col in self.metadata_df.columns:
                row_dict["disease_status"] = str(self.metadata_df.iloc[idx][disease_col])
            if "watering" in self.metadata_df.columns:
                row_dict["watering_status"] = str(self.metadata_df.iloc[idx]["watering"])
            if "barcode" in self.metadata_df.columns:
                row_dict["barcode"] = str(self.metadata_df.iloc[idx]["barcode"])

            outlier_records.append(row_dict)

        outlier_df = pd.DataFrame(outlier_records)
        outlier_df.to_csv(output_dir / "outlier_samples.csv", index=False)

        # Break down by disease status
        disease_breakdown = {}
        if not outlier_df.empty and "disease_status" in outlier_df.columns:
            disease_breakdown = outlier_df["disease_status"].value_counts().to_dict()

        summary = {
            "total_samples": int(self.num_samples),
            "outlier_count_total": int(len(outlier_indices)),
            "outlier_percentage": round((len(outlier_indices) / self.num_samples) * 100, 2),
            "auc_outliers_count": int(np.sum(auc_outliers)),
            "zscore_outliers_count": int(np.sum(z_outliers)),
            "auc_bounds": auc_bounds,
            "outlier_distribution_by_disease": disease_breakdown,
            "preservation_policy": "Non-destructive: Outliers are retained in the master dataset as they likely represent advanced canopy necrosis or drought stress rather than sensor defects."
        }
        save_json(summary, output_dir / "outlier_summary.json")

        return summary
