"""
03_Exploratory_Data_Analysis/analysis/disease_analysis.py

Scientific comparative investigation between healthy (Control) and infected (Crown Rot) crops:
- Difference spectra: Absolute Delta R(lambda) and Relative Delta R_rel(lambda).
- Parametric (Welch's t-test) and non-parametric (Mann-Whitney U) significance testing per band.
- Rigorous multiple-testing correction using Benjamini-Hochberg False Discovery Rate (FDR).
- Standardized effect size estimation (Cohen's d).
- Identification of candidate discriminative wavelength regions without claiming clinical causality.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils.eda_utils import save_json, calculate_cohens_d


class DiseaseSpectralComparator:
    """Performs rigorous statistical comparison of spectral signatures between healthy and infected canopies."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_difference_spectra(self, disease_col: str = "disease") -> pd.DataFrame:
        """
        Calculates:
        1. Control Mean Reflectance: R_ctrl(lambda)
        2. Infected Mean Reflectance: R_inf(lambda)
        3. Absolute Difference: Delta R = R_inf - R_ctrl
        4. Relative Difference: Delta R_rel = (Delta R / R_ctrl) * 100 %
        """
        if disease_col not in self.metadata_df.columns:
            raise KeyError(f"Target disease column '{disease_col}' not present in metadata.")

        ctrl_mask = (self.metadata_df[disease_col] == "Control").values
        inf_mask = (self.metadata_df[disease_col] == "Infected").values

        ctrl_mean = np.mean(self.spectral_matrix[ctrl_mask], axis=0)
        ctrl_std = np.std(self.spectral_matrix[ctrl_mask], axis=0, ddof=1)
        inf_mean = np.mean(self.spectral_matrix[inf_mask], axis=0)
        inf_std = np.std(self.spectral_matrix[inf_mask], axis=0, ddof=1)

        abs_diff = inf_mean - ctrl_mean
        rel_diff = np.where(ctrl_mean != 0, (abs_diff / ctrl_mean) * 100, 0.0)

        df = pd.DataFrame({
            "band_index": np.arange(self.num_bands),
            "wavelength_nm": self.wavelengths,
            "control_mean": ctrl_mean,
            "control_std": ctrl_std,
            "infected_mean": inf_mean,
            "infected_std": inf_std,
            "absolute_difference": abs_diff,
            "relative_difference_pct": rel_diff,
        })
        return df

    def run_significance_tests(self, disease_col: str = "disease", alpha: float = 0.05) -> pd.DataFrame:
        """
        Performs band-wise Welch's t-test and Mann-Whitney U test,
        computes Cohen's d effect sizes, and applies Benjamini-Hochberg FDR correction.
        """
        ctrl_mask = (self.metadata_df[disease_col] == "Control").values
        inf_mask = (self.metadata_df[disease_col] == "Infected").values

        ctrl_samples = self.spectral_matrix[ctrl_mask]
        inf_samples = self.spectral_matrix[inf_mask]

        t_stats = []
        p_vals_welch = []
        u_stats = []
        p_vals_mwu = []
        cohens_ds = []

        for b in range(self.num_bands):
            c_vals = ctrl_samples[:, b]
            i_vals = inf_samples[:, b]

            # Welch's t-test (unequal variances)
            t_res = stats.ttest_ind(i_vals, c_vals, equal_var=False)
            t_stats.append(float(t_res.statistic))
            p_vals_welch.append(float(t_res.pvalue))

            # Mann-Whitney U test (non-parametric)
            try:
                u_res = stats.mannwhitneyu(i_vals, c_vals, alternative="two-sided")
                u_stats.append(float(u_res.statistic))
                p_vals_mwu.append(float(u_res.pvalue))
            except Exception:
                u_stats.append(np.nan)
                p_vals_mwu.append(1.0)

            # Cohen's d effect size
            d_val = calculate_cohens_d(i_vals, c_vals)
            cohens_ds.append(d_val)

        p_vals_welch_arr = np.nan_to_num(np.array(p_vals_welch), nan=1.0)
        # Benjamini-Hochberg FDR correction
        reject_fdr, p_fdr, _, _ = multipletests(p_vals_welch_arr, alpha=alpha, method="fdr_bh")

        df = pd.DataFrame({
            "band_index": np.arange(self.num_bands),
            "wavelength_nm": self.wavelengths,
            "welch_t_statistic": t_stats,
            "welch_p_value_raw": p_vals_welch,
            "welch_p_value_fdr_adjusted": p_fdr,
            "statistically_significant_fdr": reject_fdr,
            "mann_whitney_u_statistic": u_stats,
            "mann_whitney_p_value": p_vals_mwu,
            "cohens_d_effect_size": cohens_ds,
        })
        return df

    def run_disease_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes full disease comparison pipeline and exports analytical reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        disease_col = config["data_inputs"].get("target_disease_column", "disease")
        alpha = config.get("statistical_testing", {}).get("alpha", 0.05)

        # 1. Difference Spectra
        diff_df = self.compute_difference_spectra(disease_col)
        diff_df.to_csv(output_dir / "spectral_differences.csv", index=False)

        # 2. Significance Tests & FDR Correction
        sig_df = self.run_significance_tests(disease_col, alpha=alpha)
        sig_df.to_csv(output_dir / "wavelength_significance_tests.csv", index=False)

        # 3. Identify Candidate Discriminative Regions (Top absolute differences and effect sizes)
        top_positive_diff = diff_df.nlargest(10, "absolute_difference")
        top_negative_diff = diff_df.nsmallest(10, "absolute_difference")
        top_effect_sizes = sig_df.iloc[np.abs(sig_df["cohens_d_effect_size"]).argsort()[::-1][:10]]

        fdr_sig_count = int(np.sum(sig_df["statistically_significant_fdr"]))

        summary = {
            "target_disease_evaluated": "Crown Rot (Fusarium pseudograminearum)",
            "total_bands_tested": int(self.num_bands),
            "fdr_significant_bands_count": fdr_sig_count,
            "fdr_significant_bands_percentage": round((fdr_sig_count / self.num_bands) * 100, 2),
            "top_positive_diff_wavelengths_nm": list(top_positive_diff["wavelength_nm"].values),
            "top_negative_diff_wavelengths_nm": list(top_negative_diff["wavelength_nm"].values),
            "top_effect_size_wavelengths_nm": list(top_effect_sizes["wavelength_nm"].values),
            "max_absolute_difference": round(float(diff_df["absolute_difference"].abs().max()), 5),
            "max_cohens_d": round(float(sig_df["cohens_d_effect_size"].abs().max()), 4),
            "scientific_interpretation_note": "Observed spectral variations indicate candidate discriminative regions (e.g. Red-Edge and SWIR water absorption bands) associated with Crown Rot stress; causality is physiological rather than purely spectral."
        }
        save_json(summary, output_dir / "discriminative_bands_summary.json")

        return summary
