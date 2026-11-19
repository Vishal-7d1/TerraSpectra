"""
04_Feature_Engineering/extraction/derivative_features.py

Numerical spectral differentiation and Red-Edge Position (REP) engineering:
- Computes 1st derivative (dR/dlambda) and 2nd derivative (d^2R/dlambda^2) across non-uniform bands.
- Extracts Red-Edge inflection dynamics (REP wavelength, amplitude, integrated derivative area).
- Quantifies spectral slope entering liquid water absorption bands (1450 nm & 1940 nm).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


class SpectralDerivativeExtractor:
    """Extracts first and second derivative morphological features from continuous reflectance."""

    def __init__(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def compute_continuous_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates central differences taking non-uniform wavelength intervals into account:
        d1[i] = (R[i+1] - R[i-1]) / (wl[i+1] - wl[i-1])
        """
        d1 = np.gradient(self.spectral_matrix, self.wavelengths, axis=1)
        d2 = np.gradient(d1, self.wavelengths, axis=1)
        return d1, d2

    def extract_derivative_features(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extracts key morphological derivative metrics per observation."""
        d1, d2 = self.compute_continuous_derivatives()

        # 1. Red-Edge Mask (680 nm to 750 nm)
        re_mask = (self.wavelengths >= 680.0) & (self.wavelengths <= 750.0)
        re_wls = self.wavelengths[re_mask]
        re_d1 = d1[:, re_mask]
        re_d2 = d2[:, re_mask]

        # Red-Edge Position (REP): Wavelength where 1st derivative is maximized
        rep_indices = np.argmax(re_d1, axis=1)
        rep_wavelengths = re_wls[rep_indices]
        rep_amplitudes = np.max(re_d1, axis=1)
        rep_areas = trapezoid(re_d1, re_wls, axis=1)

        # 2. Green reflectance slope (520 nm to 560 nm)
        green_mask = (self.wavelengths >= 520.0) & (self.wavelengths <= 560.0)
        green_slope = np.mean(d1[:, green_mask], axis=1)

        # 3. SWIR water absorption derivative slopes (~1450 nm and ~1940 nm)
        w1450_mask = (self.wavelengths >= 1400.0) & (self.wavelengths <= 1460.0)
        w1940_mask = (self.wavelengths >= 1900.0) & (self.wavelengths <= 1950.0)
        d1_w1450 = np.mean(d1[:, w1450_mask], axis=1)
        d1_w1940 = np.mean(d1[:, w1940_mask], axis=1)

        # 4. 2nd derivative curvature metrics in Red-Edge
        d2_re_min = np.min(re_d2, axis=1)
        d2_re_max = np.max(re_d2, axis=1)

        df = pd.DataFrame({
            "DERIV_REP_wavelength_nm": rep_wavelengths,
            "DERIV_REP_amplitude": rep_amplitudes,
            "DERIV_REP_area": rep_areas,
            "DERIV_green_slope": green_slope,
            "DERIV_swir_water_slope_1450": d1_w1450,
            "DERIV_swir_water_slope_1940": d1_w1940,
            "DERIV_red_edge_min_curvature": d2_re_min,
            "DERIV_red_edge_max_curvature": d2_re_max,
        })

        metadata = {
            "features_extracted": list(df.columns),
            "red_edge_window_nm": [680.0, 750.0],
            "water_absorption_windows_nm": [[1400.0, 1460.0], [1900.0, 1950.0]],
        }
        return df, metadata
