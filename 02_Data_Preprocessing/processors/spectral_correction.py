"""
02_Data_Preprocessing/processors/spectral_correction.py

Implements chemometric normalization techniques and atmospheric water-absorption band masking.
Includes strict Fit/Transform architecture to prevent data leakage.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import joblib
from pathlib import Path

from utils.preprocessing_utils import setup_logger

logger = setup_logger("spectral_correction")


class WaterAbsorptionMasker:
    """
    Identifies and optionally masks strong water vapor and liquid water absorption bands
    (~1350-1420nm and ~1800-1950nm) where atmospheric attenuation creates severe noise.
    """

    def __init__(self, intervals: Optional[List[List[float]]] = None):
        self.intervals = intervals or [[1350.0, 1420.0], [1800.0, 1950.0]]

    def get_mask(self, wavelengths: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True = valid band, False = masked water band."""
        mask = np.ones(len(wavelengths), dtype=bool)
        for low, high in self.intervals:
            in_range = (wavelengths >= low) & (wavelengths <= high)
            mask[in_range] = False
        return mask

    def apply(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Filters out masked water bands."""
        valid_mask = self.get_mask(wavelengths)
        filtered_spectra = spectral_matrix[:, valid_mask]
        filtered_wls = wavelengths[valid_mask]
        masked_count = int((~valid_mask).sum())

        report = {
            "original_band_count": len(wavelengths),
            "masked_band_count": masked_count,
            "retained_band_count": len(filtered_wls),
            "masked_intervals_nm": self.intervals
        }
        logger.info(f"Water absorption filter: masked {masked_count} bands, retained {len(filtered_wls)} bands.")
        return filtered_spectra, filtered_wls, report


class SpectralNormalizer:
    """
    Normalizes hyperspectral reflectance. Supports:
    - 'snv': Standard Normal Variate (computed per-sample, zero data leakage)
    - 'standard': Band-wise Z-score with strict Fit/Transform separation
    - 'min_max': Band-wise Min-Max scaling with strict Fit/Transform separation
    - 'vector': L2 Euclidean vector normalization
    """

    def __init__(self, method: str = "snv"):
        self.method = method.lower()
        self.fitted_ = False
        self.params_: Dict[str, Any] = {}

    def fit(self, X: np.ndarray) -> "SpectralNormalizer":
        """
        Learns normalization parameters strictly from training data X.
        """
        if self.method == "standard":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0.0] = 1.0  # Handle zero-variance bands
            self.params_ = {"mean": mean, "std": std}
        elif self.method == "min_max":
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            diff = max_val - min_val
            diff[diff == 0.0] = 1.0
            self.params_ = {"min": min_val, "max": max_val, "diff": diff}
        elif self.method in ["snv", "vector"]:
            # Sample-wise transformations do not depend on inter-sample statistics
            self.params_ = {"sample_wise": True}
        else:
            raise ValueError(f"Unsupported normalization method: '{self.method}'")

        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms input matrix X using learned parameters (prevents data leakage).
        """
        if not self.fitted_ and self.method not in ["snv", "vector"]:
            raise RuntimeError("SpectralNormalizer must be fitted on training data before calling transform()!")

        X_norm = X.copy()

        if self.method == "snv":
            # SNV: (spectrum - mean_spectrum) / std_spectrum
            mean_sample = np.mean(X_norm, axis=1, keepdims=True)
            std_sample = np.std(X_norm, axis=1, keepdims=True)
            std_sample[std_sample == 0.0] = 1e-6
            return (X_norm - mean_sample) / std_sample

        elif self.method == "vector":
            # L2 norm
            l2_norms = np.linalg.norm(X_norm, axis=1, keepdims=True)
            l2_norms[l2_norms == 0.0] = 1e-6
            return X_norm / l2_norms

        elif self.method == "standard":
            return (X_norm - self.params_["mean"]) / self.params_["std"]

        elif self.method == "min_max":
            return (X_norm - self.params_["min"]) / self.params_["diff"]

        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits on X and transforms X."""
        return self.fit(X).transform(X)

    def save(self, file_path: Path) -> None:
        """Persists fitted normalizer object."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Persisted fitted SpectralNormalizer to: {file_path}")

    @staticmethod
    def load(file_path: Path) -> "SpectralNormalizer":
        """Loads a persisted normalizer."""
        return joblib.load(file_path)
