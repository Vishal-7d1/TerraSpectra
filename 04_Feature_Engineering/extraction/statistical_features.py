"""
04_Feature_Engineering/extraction/statistical_features.py

Statistical moments and domain-stratified distribution engineering:
- Global moments across entire 450 - 2400 nm continuum.
- Regional moments across 5 biophysical spectral regions:
  1. Visible (VIS): 450 - 700 nm
  2. Red-Edge (RE): 680 - 750 nm
  3. Near-Infrared (NIR): 750 - 1300 nm
  4. Short-Wave Infrared 1 (SWIR1): 1300 - 1900 nm
  5. Short-Wave Infrared 2 (SWIR2): 1900 - 2400 nm
"""

from typing import Dict, Any, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalFeatureExtractor:
    """Computes global and domain-specific statistical features."""

    def __init__(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def extract_statistical_features(self, config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculates 22 global and domain-specific statistical features."""
        df = pd.DataFrame(index=np.arange(self.num_samples))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 1. Global moments across all bands
            g_means = np.mean(self.spectral_matrix, axis=1)
            g_stds = np.std(self.spectral_matrix, axis=1, ddof=1)
            g_medians = np.median(self.spectral_matrix, axis=1)
            g_skews = stats.skew(self.spectral_matrix, axis=1)
            g_kurts = stats.kurtosis(self.spectral_matrix, axis=1)
            g_q25 = np.percentile(self.spectral_matrix, 25, axis=1)
            g_q75 = np.percentile(self.spectral_matrix, 75, axis=1)
            g_iqr = g_q75 - g_q25
            g_cv = np.where(g_means != 0, g_stds / g_means, 0.0)

            df["STAT_global_mean"] = g_means
            df["STAT_global_std"] = g_stds
            df["STAT_global_median"] = g_medians
            df["STAT_global_skewness"] = g_skews
            df["STAT_global_kurtosis"] = g_kurts
            df["STAT_global_iqr"] = g_iqr
            df["STAT_global_cv"] = g_cv

            # 2. Regional moments
            regions = {
                "vis": (450.0, 700.0),
                "re": (680.0, 750.0),
                "nir": (750.0, 1300.0),
                "swir1": (1300.0, 1900.0),
                "swir2": (1900.0, 2400.0),
            }

            for reg_name, (w_min, w_max) in regions.items():
                mask = (self.wavelengths >= w_min) & (self.wavelengths <= w_max)
                if np.sum(mask) == 0:
                    continue
                sub = self.spectral_matrix[:, mask]
                r_mean = np.mean(sub, axis=1)
                r_std = np.std(sub, axis=1, ddof=1)
                r_q25 = np.percentile(sub, 25, axis=1)
                r_q75 = np.percentile(sub, 75, axis=1)

                df[f"STAT_{reg_name}_mean"] = r_mean
                df[f"STAT_{reg_name}_std"] = r_std
                df[f"STAT_{reg_name}_iqr"] = r_q75 - r_q25

            # Clean any stray nan/inf
            for col in df.columns:
                df[col] = np.nan_to_num(df[col].values, nan=0.0, posinf=0.0, neginf=0.0)

        metadata = {
            "features_extracted": list(df.columns),
            "total_statistical_features": len(df.columns),
        }
        return df, metadata
