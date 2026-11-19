"""
04_Feature_Engineering/extraction/statistical_features.py

Sample-level statistical moment and regional dispersion feature engineering:
- Global moments (mean, std, median, skewness, kurtosis, IQR, CV).
- Regional moments partitioned across the 5 biophysical domains (VIS, Red-Edge, NIR, SWIR-1, SWIR-2).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalFeatureExtractor:
    """Computes global and biophysical regional distribution metrics per observation."""

    def __init__(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def extract_statistical_features(self, config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculates 22 global and domain-specific statistical features."""
        df = pd.DataFrame(index=np.arange(self.num_samples))

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

        metadata = {
            "features_extracted": list(df.columns),
            "total_statistical_features": len(df.columns),
        }
        return df, metadata
