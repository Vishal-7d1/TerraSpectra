"""
03_Exploratory_Data_Analysis/analysis/statistical_analysis.py

Comprehensive parametric and non-parametric statistical exploratory analysis
across spectral bands, individual plant observations, and experimental treatments.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats

from utils.eda_utils import save_json, calculate_auc


class SpectralStatisticalAnalyzer:
    """Computes band-wise, observation-wise, and factor-stratified descriptive statistics."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_band_statistics(self) -> pd.DataFrame:
        """
        Computes 12 key descriptive metrics across every continuous spectral band:
        mean, median, std, variance, min, max, range, Q1, Q3, IQR, skewness, kurtosis, and CV.
        """
        means = np.mean(self.spectral_matrix, axis=0)
        medians = np.median(self.spectral_matrix, axis=0)
        stds = np.std(self.spectral_matrix, axis=0, ddof=1)
        variances = np.var(self.spectral_matrix, axis=0, ddof=1)
        mins = np.min(self.spectral_matrix, axis=0)
        maxs = np.max(self.spectral_matrix, axis=0)
        ranges = maxs - mins
        q25 = np.percentile(self.spectral_matrix, 25, axis=0)
        q75 = np.percentile(self.spectral_matrix, 75, axis=0)
        iqr = q75 - q25
        skews = stats.skew(self.spectral_matrix, axis=0)
        kurts = stats.kurtosis(self.spectral_matrix, axis=0)
        cv = np.where(means != 0, stds / means, 0.0)

        df = pd.DataFrame({
            "band_index": np.arange(self.num_bands),
            "wavelength_nm": self.wavelengths,
            "mean": means,
            "median": medians,
            "std": stds,
            "variance": variances,
            "min": mins,
            "max": maxs,
            "range": ranges,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "skewness": skews,
            "kurtosis": kurts,
            "coefficient_of_variation": cv,
        })
        return df

    def compute_grouped_band_statistics(self, group_col: str) -> pd.DataFrame:
        """Computes mean and std across bands grouped by an experimental factor (e.g. disease)."""
        if group_col not in self.metadata_df.columns:
            return pd.DataFrame()

        unique_groups = self.metadata_df[group_col].unique()
        records = []

        for idx, wl in enumerate(self.wavelengths):
            rec = {"band_index": idx, "wavelength_nm": wl}
            for grp in unique_groups:
                mask = (self.metadata_df[group_col] == grp).values
                grp_vals = self.spectral_matrix[mask, idx]
                rec[f"{grp}_mean"] = float(np.mean(grp_vals))
                rec[f"{grp}_std"] = float(np.std(grp_vals, ddof=1))
                rec[f"{grp}_median"] = float(np.median(grp_vals))
            records.append(rec)

        return pd.DataFrame(records)

    def compute_sample_level_metrics(self) -> pd.DataFrame:
        """
        Computes observation-level summary statistics:
        mean reflectance, std, max, min, and Area Under Spectral Curve (AUC).
        """
        sample_means = np.mean(self.spectral_matrix, axis=1)
        sample_stds = np.std(self.spectral_matrix, axis=1, ddof=1)
        sample_maxs = np.max(self.spectral_matrix, axis=1)
        sample_mins = np.min(self.spectral_matrix, axis=1)
        # Trapezoidal numerical integration for AUC across wavelengths
        sample_auc = calculate_auc(self.spectral_matrix, self.wavelengths, axis=1)

        sample_stats_df = pd.DataFrame({
            "sample_index": np.arange(self.num_samples),
            "sample_mean_reflectance": sample_means,
            "sample_std_reflectance": sample_stds,
            "sample_min_reflectance": sample_mins,
            "sample_max_reflectance": sample_maxs,
            "spectral_auc": sample_auc,
        })
        return sample_stats_df

    def run_statistical_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes full statistical suite and saves reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Band-wise Statistics
        band_df = self.compute_band_statistics()
        band_df.to_csv(output_dir / "band_statistics.csv", index=False)

        # 2. Disease-stratified Band Statistics
        disease_col = config["data_inputs"].get("target_disease_column", "disease")
        disease_stats_df = self.compute_grouped_band_statistics(disease_col)
        disease_stats_df.to_csv(output_dir / "disease_grouped_statistics.csv", index=False)

        # 3. Watering-stratified Band Statistics
        watering_col = config["data_inputs"].get("watering_column", "watering")
        watering_stats_df = self.compute_grouped_band_statistics(watering_col)
        if not watering_stats_df.empty:
            watering_stats_df.to_csv(output_dir / "watering_grouped_statistics.csv", index=False)

        # 4. Sample-level Statistics
        sample_df = self.compute_sample_level_metrics()
        sample_df.to_csv(output_dir / "sample_level_statistics.csv", index=False)

        # 5. Global Summary Metrics
        summary = {
            "global_reflectance_mean": round(float(np.mean(self.spectral_matrix)), 5),
            "global_reflectance_std": round(float(np.std(self.spectral_matrix)), 5),
            "global_reflectance_min": round(float(np.min(self.spectral_matrix)), 5),
            "global_reflectance_max": round(float(np.max(self.spectral_matrix)), 5),
            "highest_variance_band_nm": float(band_df.loc[band_df["variance"].idxmax(), "wavelength_nm"]),
            "lowest_variance_band_nm": float(band_df.loc[band_df["variance"].idxmin(), "wavelength_nm"]),
            "highest_cv_band_nm": float(band_df.loc[band_df["coefficient_of_variation"].idxmax(), "wavelength_nm"]),
            "lowest_cv_band_nm": float(band_df.loc[band_df["coefficient_of_variation"].idxmin(), "wavelength_nm"]),
        }
        save_json(summary, output_dir / "statistical_summary.json")

        return summary
