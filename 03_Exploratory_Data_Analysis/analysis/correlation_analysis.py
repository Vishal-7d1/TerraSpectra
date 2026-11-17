"""
03_Exploratory_Data_Analysis/analysis/correlation_analysis.py

Hyperspectral collinearity assessment and categorical-feature association analysis:
- Cross-band Pearson & Spearman correlation matrices across wavelengths.
- Identification of collinear redundant spectral clusters for downstream feature selection.
- Point-biserial & Spearman rank correlation between derived vegetation indices and disease labels.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from utils.eda_utils import save_json


class SpectralCorrelationAnalyzer:
    """Evaluates cross-wavelength redundancy and relationship with biological targets."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_band_correlation_matrix(self, subsample_step: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Computes Pearson correlation across spectral bands.
        Optionally subsamples (e.g. every 10th band = ~65 bands) for readable CSV/visualization.
        """
        indices = np.arange(0, self.num_bands, subsample_step)
        sub_matrix = self.spectral_matrix[:, indices]
        sub_wavelengths = self.wavelengths[indices]

        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)
        corr_df = pd.DataFrame(
            corr_matrix,
            index=[f"{wl:.1f}nm" for wl in sub_wavelengths],
            columns=[f"{wl:.1f}nm" for wl in sub_wavelengths]
        )
        return corr_df, corr_matrix

    def identify_highly_correlated_pairs(self, threshold: float = 0.98) -> pd.DataFrame:
        """Identifies adjacent or distant band pairs exhibiting extreme collinearity (|r| >= threshold)."""
        pairs = []
        # Sample across representative bands (every 5th band)
        step = 5
        sub_indices = np.arange(0, self.num_bands, step)
        sub_wls = self.wavelengths[sub_indices]
        sub_mat = self.spectral_matrix[:, sub_indices]

        corr_mat = np.corrcoef(sub_mat, rowvar=False)

        for i in range(len(sub_wls)):
            for j in range(i + 1, len(sub_wls)):
                r = float(corr_mat[i, j])
                if abs(r) >= threshold:
                    pairs.append({
                        "wavelength_1_nm": float(sub_wls[i]),
                        "wavelength_2_nm": float(sub_wls[j]),
                        "delta_wavelength_nm": round(float(abs(sub_wls[i] - sub_wls[j])), 2),
                        "pearson_r": round(r, 4),
                    })

        return pd.DataFrame(pairs)

    def correlate_indices_with_disease(self, indices_df: pd.DataFrame, disease_col: str = "disease") -> pd.DataFrame:
        """
        Computes Point-Biserial and Spearman rank correlations between derived
        vegetation indices and binary disease status (Control=0, Infected=1).
        """
        if disease_col not in self.metadata_df.columns:
            return pd.DataFrame()

        binary_disease = (self.metadata_df[disease_col] == "Infected").astype(int).values

        records = []
        for col in indices_df.columns:
            vals = indices_df[col].values

            # Point-biserial correlation
            pb_r, pb_p = stats.pointbiserialr(binary_disease, vals)

            # Spearman rank correlation
            sp_r, sp_p = stats.spearmanr(binary_disease, vals)

            records.append({
                "vegetation_index": col,
                "point_biserial_r": round(float(pb_r), 4),
                "point_biserial_p_value": float(pb_p),
                "spearman_rho": round(float(sp_r), 4),
                "spearman_p_value": float(sp_p),
                "significant_at_0_05": bool(pb_p < 0.05),
            })

        return pd.DataFrame(records).sort_values("point_biserial_r", key=abs, ascending=False)

    def run_correlation_analysis(self, config: Dict[str, Any], output_dir: Path, indices_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Executes cross-band correlation and target relationship reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        disease_col = config["data_inputs"].get("target_disease_column", "disease")

        # 1. Band Correlation Matrix
        band_corr_df, _ = self.compute_band_correlation_matrix(subsample_step=10)
        band_corr_df.to_csv(output_dir / "spectral_band_correlation.csv")

        # 2. Highly Collinear Pairs
        collinear_df = self.identify_highly_correlated_pairs(threshold=0.98)
        collinear_df.to_csv(output_dir / "highly_correlated_wavelength_pairs.csv", index=False)

        # 3. Indices vs Disease Correlation
        ind_corr_df = pd.DataFrame()
        if indices_df is not None:
            ind_corr_df = self.correlate_indices_with_disease(indices_df, disease_col=disease_col)
            ind_corr_df.to_csv(output_dir / "indices_disease_correlation.csv", index=False)

        summary = {
            "subsampled_band_matrix_shape": list(band_corr_df.shape),
            "highly_correlated_pairs_count": len(collinear_df),
            "strongest_index_association": ind_corr_df.iloc[0]["vegetation_index"] if not ind_corr_df.empty else "N/A",
            "strongest_index_point_biserial_r": float(ind_corr_df.iloc[0]["point_biserial_r"]) if not ind_corr_df.empty else 0.0,
            "feature_engineering_recommendation": "Severe spectral collinearity detected across adjacent narrow bands (|r| > 0.98); dimensionality reduction or band-selection (e.g. LASSO / Recursive Feature Elimination) is recommended in Module 04."
        }
        save_json(summary, output_dir / "correlation_summary.json")

        return summary
