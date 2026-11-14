"""
02_Data_Preprocessing/processors/noise_reduction.py

Implements noise reduction algorithms for hyperspectral reflectance signatures:
1. Savitzky-Golay filtering (preserves narrow biochemical absorption peaks and Red-Edge inflection)
2. 1D Median filtering (removes single-band spike noise)
3. Moving average smoothing (baseline comparator)
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import savgol_filter, medfilt

from utils.preprocessing_utils import setup_logger

logger = setup_logger("noise_reduction")


class HyperspectralNoiseReducer:
    """
    Applies chemometric and bio-optical noise reduction across spectral wavelength channels.
    """

    def __init__(
        self,
        method: str = "savitzky_golay",
        window_length: int = 11,
        polyorder: int = 2,
    ):
        self.method = method.lower()
        self.window_length = window_length if window_length % 2 != 0 else window_length + 1
        self.polyorder = polyorder

    def estimate_noise_level(self, spectra: np.ndarray) -> float:
        """
        Estimates spectral sensor noise standard deviation using second-order differences.
        Based on pseudo-residual estimation: sigma ~ std(diff(diff(y))) / sqrt(6)
        """
        if spectra.shape[1] < 3:
            return 0.0
        d2 = np.diff(spectra, n=2, axis=1)
        sigma = float(np.std(d2) / np.sqrt(6))
        return round(sigma, 6)

    def reduce_noise(self, spectral_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies configured noise reduction technique.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (denoised_matrix, noise_report)
        """
        initial_noise = self.estimate_noise_level(spectral_matrix)
        denoised = spectral_matrix.copy()

        logger.info(f"Applying noise reduction (Method: {self.method}, Window: {self.window_length})...")

        if self.method == "savitzky_golay":
            if self.window_length <= self.polyorder:
                raise ValueError(
                    f"Savitzky-Golay window_length ({self.window_length}) must be strictly greater than polyorder ({self.polyorder})."
                )
            denoised = savgol_filter(
                spectral_matrix,
                window_length=self.window_length,
                polyorder=self.polyorder,
                axis=1
            )
        elif self.method == "median":
            # Apply median filter along axis 1 (across spectral bands)
            denoised = medfilt(spectral_matrix, kernel_size=(1, self.window_length))
        elif self.method == "moving_average":
            kernel = np.ones(self.window_length) / self.window_length
            denoised = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=spectral_matrix
            )
        else:
            raise ValueError(f"Unsupported noise reduction method: '{self.method}'")

        final_noise = self.estimate_noise_level(denoised)
        noise_reduction_ratio = round((1 - (final_noise / (initial_noise + 1e-8))) * 100, 2)

        report = {
            "method": self.method,
            "window_length": self.window_length,
            "polyorder": self.polyorder if self.method == "savitzky_golay" else None,
            "initial_noise_proxy_sigma": initial_noise,
            "final_noise_proxy_sigma": final_noise,
            "noise_reduction_percentage": noise_reduction_ratio,
        }

        logger.info(
            f"Noise reduction complete: initial noise={initial_noise:.5f}, "
            f"final noise={final_noise:.5f} ({noise_reduction_ratio}% reduction)."
        )

        return denoised, report
