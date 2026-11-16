"""
02_Data_Preprocessing/processors/missing_value_handler.py

Detects and handles missing (NaN), infinite, and invalid reflectance measurements.
Implements physics-based linear spectral interpolation along contiguous wavelength bands.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from utils.preprocessing_utils import setup_logger

logger = setup_logger("missing_value_handler")


class MissingValueHandler:
    """
    Detects and treats missing, infinite, and out-of-range spectral data.
    """

    def __init__(
        self,
        strategy: str = "linear_spectral_interpolation",
        min_allowed: float = 0.0,
        max_allowed: float = 1.0,
    ):
        self.strategy = strategy
        self.min_allowed = min_allowed
        self.max_allowed = max_allowed

    def audit(self, spectral_matrix: np.ndarray) -> Dict[str, Any]:
        """Performs non-destructive diagnostic check on spectral matrix."""
        total_elements = spectral_matrix.size
        nan_count = int(np.isnan(spectral_matrix).sum())
        inf_count = int(np.isinf(spectral_matrix).sum())
        finite_mask = np.isfinite(spectral_matrix)
        negative_count = int((finite_mask & (spectral_matrix < self.min_allowed)).sum())
        over_max_count = int((finite_mask & (spectral_matrix > self.max_allowed)).sum())

        return {
            "total_elements": total_elements,
            "nan_count": nan_count,
            "nan_percentage": round((nan_count / total_elements) * 100, 4),
            "inf_count": inf_count,
            "negative_count": negative_count,
            "over_max_count": over_max_count,
            "is_clean": (nan_count == 0 and inf_count == 0 and negative_count == 0 and over_max_count == 0),
        }

    def clean(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cleans NaNs, Infs, and clips invalid reflectance to physically valid bounds.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (cleaned_matrix, cleaning_audit_report)
        """
        cleaned = spectral_matrix.copy()
        audit_before = self.audit(cleaned)

        if audit_before["is_clean"]:
            logger.info("Spectral matrix is already completely clean (0 NaNs, 0 Infs, valid reflectance range).")
            return cleaned, {"actions_taken": ["No missing/invalid values detected."], "audit_before": audit_before}

        actions_taken = []

        # Replace Infs with NaN first for unified handling
        if audit_before["inf_count"] > 0:
            cleaned[np.isinf(cleaned)] = np.nan
            actions_taken.append(f"Replaced {audit_before['inf_count']} infinite values with NaN.")

        # Handle NaNs
        if np.isnan(cleaned).any():
            if self.strategy == "linear_spectral_interpolation":
                # Interpolate across wavelengths per sample
                for row_idx in range(cleaned.shape[0]):
                    row = cleaned[row_idx, :]
                    if np.isnan(row).any():
                        nans = np.isnan(row)
                        valid_x = wavelengths[~nans]
                        valid_y = row[~nans]
                        if len(valid_x) >= 2:
                            row[nans] = np.interp(wavelengths[nans], valid_x, valid_y)
                        else:
                            row[nans] = np.nanmedian(cleaned)
                        cleaned[row_idx, :] = row
                actions_taken.append("Applied contiguous 1D linear spectral interpolation for NaN bands.")
            else:
                col_medians = np.nanmedian(cleaned, axis=0)
                col_medians = np.nan_to_num(col_medians, nan=0.0)
                inds = np.where(np.isnan(cleaned))
                cleaned[inds] = np.take(col_medians, inds[1])
                actions_taken.append("Applied column-wise spectral band median imputation.")

        # Clip reflectance range
        if (cleaned < self.min_allowed).any() or (cleaned > self.max_allowed).any():
            cleaned = np.clip(cleaned, self.min_allowed, self.max_allowed)
            actions_taken.append(f"Clipped values to physically valid reflectance range [{self.min_allowed}, {self.max_allowed}].")

        audit_after = self.audit(cleaned)
        logger.info(f"Missing value handling complete: {actions_taken}")

        return cleaned, {
            "actions_taken": actions_taken,
            "audit_before": audit_before,
            "audit_after": audit_after,
        }
