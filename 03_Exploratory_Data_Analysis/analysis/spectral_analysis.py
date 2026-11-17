"""
03_Exploratory_Data_Analysis/analysis/spectral_analysis.py

Hyperspectral profile characterization and biological spectral region investigation:
- Computes mean and standard deviation spectral signatures across treatments.
- Evaluates biophysical spectral regions (VIS, Red-Edge, NIR, SWIR-1, SWIR-2).
- Quantifies spectral variability and absorption peak positions.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from utils.eda_utils import save_json


class HyperspectralSignatureAnalyzer:
    """Analyzes continuous spectral signatures and biophysical wavelength regions."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_mean_signatures(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates mean and standard deviation reflectance across all observations,
        disease categories, watering conditions, and cultivars.
        """
        disease_col = config["data_inputs"].get("target_disease_column", "disease")
        watering_col = config["data_inputs"].get("watering_column", "watering")
        cultivar_col = config["data_inputs"].get("cultivar_column", "genotype ID")

        df = pd.DataFrame({"wavelength_nm": self.wavelengths})

        # Global
        df["global_mean"] = np.mean(self.spectral_matrix, axis=0)
        df["global_std"] = np.std(self.spectral_matrix, axis=0, ddof=1)

        # Disease stratification
        if disease_col in self.metadata_df.columns:
            for cls in self.metadata_df[disease_col].unique():
                mask = (self.metadata_df[disease_col] == cls).values
                df[f"disease_{cls}_mean"] = np.mean(self.spectral_matrix[mask], axis=0)
                df[f"disease_{cls}_std"] = np.std(self.spectral_matrix[mask], axis=0, ddof=1)

        # Watering stratification
        if watering_col in self.metadata_df.columns:
            for w in self.metadata_df[watering_col].unique():
                mask = (self.metadata_df[watering_col] == w).values
                df[f"watering_{w}_mean"] = np.mean(self.spectral_matrix[mask], axis=0)
                df[f"watering_{w}_std"] = np.std(self.spectral_matrix[mask], axis=0, ddof=1)

        # Cultivar stratification
        if cultivar_col in self.metadata_df.columns:
            for c in self.metadata_df[cultivar_col].unique():
                mask = (self.metadata_df[cultivar_col] == c).values
                df[f"cultivar_{c}_mean"] = np.mean(self.spectral_matrix[mask], axis=0)

        return df

    def analyze_spectral_regions(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Segments the continuous spectrum into physical domain regions:
        1. Visible (VIS): 450 - 700 nm (photosynthetic pigments: chlorophyll a/b, carotenoids)
        2. Red-Edge: 680 - 750 nm (steep inflection boundary of cellular chlorophyll absorption)
        3. Near-Infrared (NIR): 750 - 1300 nm (mesophyll internal cell scattering plateau)
        4. Short-Wave Infrared 1 (SWIR-1): 1300 - 1900 nm (water absorption & organic compounds)
        5. Short-Wave Infrared 2 (SWIR-2): 1900 - 2400 nm (lignin, cellulose, deep moisture bands)
        """
        regions_cfg = config.get("spectral_regions", {
            "visible": [450.0, 700.0],
            "red_edge": [680.0, 750.0],
            "near_infrared": [750.0, 1300.0],
            "shortwave_infrared_1": [1300.0, 1900.0],
            "shortwave_infrared_2": [1900.0, 2400.0],
        })

        disease_col = config["data_inputs"].get("target_disease_column", "disease")
        has_disease = disease_col in self.metadata_df.columns
        if has_disease:
            ctrl_mask = (self.metadata_df[disease_col] == "Control").values
            inf_mask = (self.metadata_df[disease_col] == "Infected").values

        rows = []
        for reg_name, (w_min, w_max) in regions_cfg.items():
            mask_wl = (self.wavelengths >= w_min) & (self.wavelengths <= w_max)
            band_count = int(np.sum(mask_wl))
            if band_count == 0:
                continue

            sub_matrix = self.spectral_matrix[:, mask_wl]
            reg_mean = float(np.mean(sub_matrix))
            reg_std = float(np.std(sub_matrix))
            reg_max = float(np.max(sub_matrix))
            reg_min = float(np.min(sub_matrix))

            row_data = {
                "region_name": reg_name,
                "wavelength_min_nm": w_min,
                "wavelength_max_nm": w_max,
                "band_count": band_count,
                "mean_reflectance": round(reg_mean, 5),
                "std_reflectance": round(reg_std, 5),
                "min_reflectance": round(reg_min, 5),
                "max_reflectance": round(reg_max, 5),
            }

            if has_disease:
                ctrl_mean = float(np.mean(self.spectral_matrix[ctrl_mask][:, mask_wl]))
                inf_mean = float(np.mean(self.spectral_matrix[inf_mask][:, mask_wl]))
                diff = inf_mean - ctrl_mean
                pct_diff = (diff / ctrl_mean * 100) if ctrl_mean != 0 else 0.0
                row_data["control_mean"] = round(ctrl_mean, 5)
                row_data["infected_mean"] = round(inf_mean, 5)
                row_data["absolute_difference"] = round(diff, 5)
                row_data["percentage_difference"] = round(pct_diff, 2)

            rows.append(row_data)

        return pd.DataFrame(rows)

    def run_spectral_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes spectral signatures and region summaries."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sig_df = self.compute_mean_signatures(config)
        sig_df.to_csv(output_dir / "mean_spectral_signatures.csv", index=False)

        reg_df = self.analyze_spectral_regions(config)
        reg_df.to_csv(output_dir / "spectral_regions_summary.csv", index=False)

        # Identify key absorption peaks and plateau maxima
        nir_mask = (self.wavelengths >= 750.0) & (self.wavelengths <= 1100.0)
        nir_peak_wl = float(self.wavelengths[nir_mask][np.argmax(np.mean(self.spectral_matrix[:, nir_mask], axis=0))])
        
        green_mask = (self.wavelengths >= 520.0) & (self.wavelengths <= 570.0)
        green_peak_wl = float(self.wavelengths[green_mask][np.argmax(np.mean(self.spectral_matrix[:, green_mask], axis=0))])

        summary = {
            "total_bands_analyzed": int(self.num_bands),
            "wavelength_span_nm": [float(self.wavelengths.min()), float(self.wavelengths.max())],
            "nir_plateau_peak_wavelength_nm": nir_peak_wl,
            "green_reflectance_peak_wavelength_nm": green_peak_wl,
            "regions_evaluated": list(reg_df["region_name"].values),
        }
        save_json(summary, output_dir / "spectral_analysis_summary.json")

        return summary
