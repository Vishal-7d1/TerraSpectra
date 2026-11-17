"""
03_Exploratory_Data_Analysis/analysis/vegetation_indices.py

Hyperspectral narrow-band vegetation indices extraction and physiological evaluation:
- Maps theoretical wavelengths to nearest empirical narrow bands.
- Calculates NDVI, NDRE, PRI, mSR705, WBI, and CRI with zero-division safeguards.
- Performs statistical hypothesis testing (Control vs Infected) on all derived indices.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from utils.eda_utils import save_json, find_nearest_wavelength, calculate_cohens_d


class VegetationIndexExtractor:
    """Calculates scientifically documented narrow-band vegetation and water indices."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_all_indices(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Computes all configured narrow-band indices:
        - NDVI: (R_800 - R_680) / (R_800 + R_680)  [Plant canopy vigor & biomass]
        - NDRE: (R_790 - R_720) / (R_790 + R_720)  [Red-Edge chlorophyll density]
        - PRI: (R_531 - R_570) / (R_531 + R_570)   [Photochemical light use efficiency]
        - mSR705: (R_750 - R_445) / (R_705 - R_445) [Modified Simple Ratio 705]
        - WBI: R_900 / R_970                       [Canopy water content]
        - CRI: (1 / R_510) - (1 / R_550)           [Carotenoid to chlorophyll ratio]
        """
        indices_df = pd.DataFrame(index=self.metadata_df.index)
        mapping_record = {}

        # 1. NDVI (800 nm, 680 nm)
        idx_800, wl_800 = find_nearest_wavelength(800.0, self.wavelengths)
        idx_680, wl_680 = find_nearest_wavelength(680.0, self.wavelengths)
        r800 = self.spectral_matrix[:, idx_800]
        r680 = self.spectral_matrix[:, idx_680]
        denom_ndvi = r800 + r680
        indices_df["NDVI"] = np.where(denom_ndvi != 0, (r800 - r680) / denom_ndvi, 0.0)
        mapping_record["NDVI"] = {
            "name": "Normalized Difference Vegetation Index",
            "formula": "(R800 - R680) / (R800 + R680)",
            "nir_target_nm": 800.0, "nir_actual_nm": wl_800,
            "red_target_nm": 680.0, "red_actual_nm": wl_680
        }

        # 2. NDRE (790 nm, 720 nm)
        idx_790, wl_790 = find_nearest_wavelength(790.0, self.wavelengths)
        idx_720, wl_720 = find_nearest_wavelength(720.0, self.wavelengths)
        r790 = self.spectral_matrix[:, idx_790]
        r720 = self.spectral_matrix[:, idx_720]
        denom_ndre = r790 + r720
        indices_df["NDRE"] = np.where(denom_ndre != 0, (r790 - r720) / denom_ndre, 0.0)
        mapping_record["NDRE"] = {
            "name": "Normalized Difference Red Edge",
            "formula": "(R790 - R720) / (R790 + R720)",
            "nir_target_nm": 790.0, "nir_actual_nm": wl_790,
            "re_target_nm": 720.0, "re_actual_nm": wl_720
        }

        # 3. PRI (531 nm, 570 nm)
        idx_531, wl_531 = find_nearest_wavelength(531.0, self.wavelengths)
        idx_570, wl_570 = find_nearest_wavelength(570.0, self.wavelengths)
        r531 = self.spectral_matrix[:, idx_531]
        r570 = self.spectral_matrix[:, idx_570]
        denom_pri = r531 + r570
        indices_df["PRI"] = np.where(denom_pri != 0, (r531 - r570) / denom_pri, 0.0)
        mapping_record["PRI"] = {
            "name": "Photochemical Reflectance Index",
            "formula": "(R531 - R570) / (R531 + R570)",
            "wl1_target_nm": 531.0, "wl1_actual_nm": wl_531,
            "wl2_target_nm": 570.0, "wl2_actual_nm": wl_570
        }

        # 4. mSR705 (750 nm, 705 nm, 450.82 nm)
        idx_750, wl_750 = find_nearest_wavelength(750.0, self.wavelengths)
        idx_705, wl_705 = find_nearest_wavelength(705.0, self.wavelengths)
        idx_445, wl_445 = find_nearest_wavelength(445.0, self.wavelengths)
        r750 = self.spectral_matrix[:, idx_750]
        r705 = self.spectral_matrix[:, idx_705]
        r445 = self.spectral_matrix[:, idx_445]
        denom_msr = r705 - r445
        indices_df["mSR705"] = np.where(denom_msr != 0, (r750 - r445) / denom_msr, 0.0)
        mapping_record["mSR705"] = {
            "name": "Modified Simple Ratio 705",
            "formula": "(R750 - R445) / (R705 - R445)",
            "r750_actual_nm": wl_750, "r705_actual_nm": wl_705, "r445_actual_nm": wl_445
        }

        # 5. WBI (900 nm, 970 nm)
        idx_900, wl_900 = find_nearest_wavelength(900.0, self.wavelengths)
        idx_970, wl_970 = find_nearest_wavelength(970.0, self.wavelengths)
        r900 = self.spectral_matrix[:, idx_900]
        r970 = self.spectral_matrix[:, idx_970]
        indices_df["WBI"] = np.where(r970 != 0, r900 / r970, 0.0)
        mapping_record["WBI"] = {
            "name": "Water Band Index",
            "formula": "R900 / R970",
            "r900_actual_nm": wl_900, "r970_actual_nm": wl_970
        }

        # 6. CRI (510 nm, 550 nm)
        idx_510, wl_510 = find_nearest_wavelength(510.0, self.wavelengths)
        idx_550, wl_550 = find_nearest_wavelength(550.0, self.wavelengths)
        r510 = self.spectral_matrix[:, idx_510]
        r550 = self.spectral_matrix[:, idx_550]
        term1 = np.where(r510 != 0, 1.0 / r510, 0.0)
        term2 = np.where(r550 != 0, 1.0 / r550, 0.0)
        indices_df["CRI"] = term1 - term2
        mapping_record["CRI"] = {
            "name": "Carotenoid Reflectance Index",
            "formula": "(1 / R510) - (1 / R550)",
            "r510_actual_nm": wl_510, "r550_actual_nm": wl_550
        }

        return indices_df, mapping_record

    def compare_indices_by_disease(self, indices_df: pd.DataFrame, disease_col: str = "disease") -> pd.DataFrame:
        """Performs descriptive and statistical tests for each index across disease categories."""
        if disease_col not in self.metadata_df.columns:
            return pd.DataFrame()

        ctrl_mask = (self.metadata_df[disease_col] == "Control").values
        inf_mask = (self.metadata_df[disease_col] == "Infected").values

        rows = []
        for col in indices_df.columns:
            ctrl_vals = indices_df.loc[ctrl_mask, col].values
            inf_vals = indices_df.loc[inf_mask, col].values

            # Statistical testing
            t_res = stats.ttest_ind(inf_vals, ctrl_vals, equal_var=False)
            u_res = stats.mannwhitneyu(inf_vals, ctrl_vals, alternative="two-sided")
            d_val = calculate_cohens_d(inf_vals, ctrl_vals)

            rows.append({
                "vegetation_index": col,
                "control_mean": round(float(np.mean(ctrl_vals)), 4),
                "control_std": round(float(np.std(ctrl_vals, ddof=1)), 4),
                "infected_mean": round(float(np.mean(inf_vals)), 4),
                "infected_std": round(float(np.std(inf_vals, ddof=1)), 4),
                "mean_difference": round(float(np.mean(inf_vals) - np.mean(ctrl_vals)), 4),
                "welch_t_statistic": round(float(t_res.statistic), 4),
                "welch_p_value": float(t_res.pvalue),
                "mann_whitney_p_value": float(u_res.pvalue),
                "cohens_d": round(d_val, 4),
                "significant_at_0_05": bool(t_res.pvalue < 0.05),
            })

        return pd.DataFrame(rows)

    def run_indices_extraction(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Calculates indices, executes statistical comparisons, and exports files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        disease_col = config["data_inputs"].get("target_disease_column", "disease")

        # Compute indices
        indices_df, mapping = self.compute_all_indices(config)

        # Merge with key metadata for downstream feature engineering
        combined_df = pd.concat([self.metadata_df, indices_df], axis=1)

        # Export CSV and Parquet
        indices_df.to_csv(output_dir / "calculated_vegetation_indices.csv", index=False)
        combined_df.to_parquet(output_dir / "vegetation_indices_with_metadata.parquet", index=False)

        # Statistical comparison
        comp_df = self.compare_indices_by_disease(indices_df, disease_col=disease_col)
        comp_df.to_csv(output_dir / "indices_statistical_comparison.csv", index=False)

        # Mapping metadata
        save_json(mapping, output_dir / "vegetation_indices_mapping.json")

        summary = {
            "indices_calculated": list(indices_df.columns),
            "indices_count": len(indices_df.columns),
            "top_discriminative_index_by_cohens_d": str(comp_df.loc[comp_df["cohens_d"].abs().idxmax(), "vegetation_index"]),
            "top_cohens_d": float(comp_df["cohens_d"].abs().max()),
        }
        save_json(summary, output_dir / "vegetation_indices_summary.json")

        return summary
