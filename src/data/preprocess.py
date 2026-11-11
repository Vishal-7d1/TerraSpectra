"""
Preprocessing Module for Hyperspectral Data.
Provides radiometric calibration, noise reduction (Savitzky-Golay filtering),
Standard Normal Variate (SNV), and water absorption band removal.
"""

from typing import List, Optional
import numpy as np
from scipy.signal import savgol_filter


def smooth_spectra(spectra: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Applies Savitzky-Golay filter to smooth hyperspectral reflectance curves.

    Parameters:
        spectra (np.ndarray): Shape (n_samples, n_bands) or (n_bands,).
        window_length (int): Length of filter window (must be odd).
        polyorder (int): Order of the polynomial.

    Returns:
        np.ndarray: Smoothed spectra.
    """
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(spectra, window_length=window_length, polyorder=polyorder, axis=-1)


def standard_normal_variate(spectra: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) transformation to remove scatter effects.

    Parameters:
        spectra (np.ndarray): Shape (n_samples, n_bands).

    Returns:
        np.ndarray: Normalized spectra.
    """
    mean = np.mean(spectra, axis=-1, keepdims=True)
    std = np.std(spectra, axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (spectra - mean) / std


def remove_noisy_bands(spectra: np.ndarray, bad_band_indices: List[int]) -> np.ndarray:
    """
    Removes atmospheric water absorption bands (e.g. ~1350-1450nm and ~1800-1950nm).

    Parameters:
        spectra (np.ndarray): Shape (..., n_bands).
        bad_band_indices (List[int]): Band indices to drop.

    Returns:
        np.ndarray: Cleaned spectra without noisy bands.
    """
    return np.delete(spectra, bad_band_indices, axis=-1)


def calibrate_reflectance(raw_dn: np.ndarray, dark_ref: np.ndarray, white_ref: np.ndarray) -> np.ndarray:
    """
    Calibrates Digital Numbers (DN) to relative reflectance:
    Reflectance = (Raw - Dark) / (White - Dark)
    """
    denominator = white_ref - dark_ref
    denominator[denominator == 0] = 1e-6
    reflectance = (raw_dn - dark_ref) / denominator
    return np.clip(reflectance, 0.0, 1.0)

