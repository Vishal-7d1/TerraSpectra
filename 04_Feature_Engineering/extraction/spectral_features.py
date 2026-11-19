"""
04_Feature_Engineering/extraction/spectral_features.py

Spectral shape, continuum curvature, and biophysical absorption depth features:
- Area Under Spectral Curve (AUC) via trapezoidal integration.
- Regional spectral slopes (VIS, Red-Edge, NIR plateau).
- Continuum-relative water absorption depths at 1450 nm and 1940 nm.
- Classic spectral reflectance ratios (NIR/Red, NIR/Green).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from utils.feature_utils import find_nearest_wavelength


class SpectralShapeExtractor:
    """Extracts geometric, continuum, and absorption depth features."""

    def __init__(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def extract_shape_features(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculates 11 geometric and absorption depth features."""
        df = pd.DataFrame(index=np.arange(self.num_samples))

        # 1. Total integrated Spectral AUC
        df["SHAPE_total_spectral_auc"] = trapezoid(self.spectral_matrix, self.wavelengths, axis=1)

        # 2. Regional slopes
        # Visible slope (450 nm to 680 nm)
        i_vis_start, w_vis_start = find_nearest_wavelength(450.8, self.wavelengths)
        i_vis_end, w_vis_end = find_nearest_wavelength(680.0, self.wavelengths)
        delta_wl_vis = w_vis_end - w_vis_start
        df["SHAPE_vis_slope"] = (self.spectral_matrix[:, i_vis_end] - self.spectral_matrix[:, i_vis_start]) / delta_wl_vis

        # Red-Edge slope (680 nm to 750 nm)
        i_re_start, w_re_start = find_nearest_wavelength(680.0, self.wavelengths)
        i_re_end, w_re_end = find_nearest_wavelength(750.0, self.wavelengths)
        delta_wl_re = w_re_end - w_re_start
        df["SHAPE_red_edge_slope"] = (self.spectral_matrix[:, i_re_end] - self.spectral_matrix[:, i_re_start]) / delta_wl_re

        # NIR Plateau mean and slope (750 nm to 1100 nm)
        nir_mask = (self.wavelengths >= 750.0) & (self.wavelengths <= 1100.0)
        df["SHAPE_nir_plateau_mean"] = np.mean(self.spectral_matrix[:, nir_mask], axis=1)

        i_nir_start, w_nir_start = find_nearest_wavelength(750.0, self.wavelengths)
        i_nir_end, w_nir_end = find_nearest_wavelength(1100.0, self.wavelengths)
        delta_wl_nir = w_nir_end - w_nir_start
        df["SHAPE_nir_plateau_slope"] = (self.spectral_matrix[:, i_nir_end] - self.spectral_matrix[:, i_nir_start]) / delta_wl_nir

        # 3. Water Absorption Depths (Continuum relative baseline)
        # 1450 nm depth: baseline between 1300 nm and 1600 nm
        i_w1450, _ = find_nearest_wavelength(1450.0, self.wavelengths)
        i_b1300, _ = find_nearest_wavelength(1300.0, self.wavelengths)
        i_b1600, _ = find_nearest_wavelength(1600.0, self.wavelengths)
        baseline_1450 = 0.5 * (self.spectral_matrix[:, i_b1300] + self.spectral_matrix[:, i_b1600])
        df["SHAPE_water_depth_1450"] = np.where(
            baseline_1450 != 0,
            1.0 - (self.spectral_matrix[:, i_w1450] / baseline_1450),
            0.0
        )

        # 1940 nm depth: baseline between 1800 nm and 2100 nm
        i_w1940, _ = find_nearest_wavelength(1940.0, self.wavelengths)
        i_b1800, _ = find_nearest_wavelength(1800.0, self.wavelengths)
        i_b2100, _ = find_nearest_wavelength(2100.0, self.wavelengths)
        baseline_1940 = 0.5 * (self.spectral_matrix[:, i_b1800] + self.spectral_matrix[:, i_b2100])
        df["SHAPE_water_depth_1940"] = np.where(
            baseline_1940 != 0,
            1.0 - (self.spectral_matrix[:, i_w1940] / baseline_1940),
            0.0
        )

        # 4. Simple Ratios
        i_800, _ = find_nearest_wavelength(800.0, self.wavelengths)
        i_680, _ = find_nearest_wavelength(680.0, self.wavelengths)
        i_550, _ = find_nearest_wavelength(550.0, self.wavelengths)
        r800 = self.spectral_matrix[:, i_800]
        r680 = self.spectral_matrix[:, i_680]
        r550 = self.spectral_matrix[:, i_550]

        df["SHAPE_sr_nir_red"] = np.where(r680 != 0, r800 / r680, 0.0)
        df["SHAPE_sr_nir_green"] = np.where(r550 != 0, r800 / r550, 0.0)

        # 5. Peak Reflectance & Wavelength
        df["SHAPE_peak_reflectance"] = np.max(self.spectral_matrix, axis=1)
        peak_indices = np.argmax(self.spectral_matrix, axis=1)
        df["SHAPE_peak_wavelength_nm"] = self.wavelengths[peak_indices]

        metadata = {
            "features_extracted": list(df.columns),
            "total_shape_features": len(df.columns),
        }
        return df, metadata
