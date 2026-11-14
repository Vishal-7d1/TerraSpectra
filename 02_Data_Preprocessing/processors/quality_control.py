"""
02_Data_Preprocessing/processors/quality_control.py

Performs comprehensive scientific validation, distortion checks,
and numerical stability audits on preprocessed hyperspectral matrices.
"""

from typing import Dict, Any
import numpy as np

from utils.preprocessing_utils import setup_logger

logger = setup_logger("quality_control")


class PreprocessingQualityControl:
    """
    Evaluates fidelity, dimension preservation, and scientific validity of preprocessed data.
    """

    def __init__(
        self,
        max_allowed_rmse: float = 0.05,
        min_correlation: float = 0.98,
    ):
        self.max_allowed_rmse = max_allowed_rmse
        self.min_correlation = min_correlation

    def compute_spectral_fidelity(self, raw_spectra: np.ndarray, processed_spectra: np.ndarray) -> Dict[str, float]:
        """
        Measures distortion between raw and smoothed spectra.
        """
        if raw_spectra.shape != processed_spectra.shape:
            # Different shapes (e.g. if bands were masked)
            return {"rmse": -1.0, "mean_correlation": -1.0, "status": "SHAPES_DIFFERENT"}

        # RMSE across all values
        rmse = float(np.sqrt(np.mean((raw_spectra - processed_spectra) ** 2)))

        # Mean sample-wise Pearson correlation
        correlations = []
        for i in range(min(len(raw_spectra), 200)):  # sample 200 rows for speed
            r = np.corrcoef(raw_spectra[i], processed_spectra[i])[0, 1]
            if np.isfinite(r):
                correlations.append(r)

        mean_corr = float(np.mean(correlations)) if correlations else 1.0

        return {
            "spectral_rmse": round(rmse, 6),
            "mean_pearson_correlation": round(mean_corr, 6),
            "fidelity_passed": bool(rmse <= self.max_allowed_rmse and mean_corr >= self.min_correlation),
        }

    def audit(
        self,
        raw_spectra: np.ndarray,
        processed_spectra: np.ndarray,
        raw_labels: np.ndarray,
        processed_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Runs full pre- and post-processing quality control comparison.
        """
        n_raw_samples, n_raw_bands = raw_spectra.shape
        n_proc_samples, n_proc_bands = processed_spectra.shape

        labels_aligned = bool(len(raw_labels) == len(processed_labels) and np.array_equal(raw_labels, processed_labels))
        nan_proc = int(np.isnan(processed_spectra).sum())
        inf_proc = int(np.isinf(processed_spectra).sum())

        fidelity = self.compute_spectral_fidelity(raw_spectra, processed_spectra)

        stats_raw = {
            "min": round(float(np.min(raw_spectra)), 4),
            "max": round(float(np.max(raw_spectra)), 4),
            "mean": round(float(np.mean(raw_spectra)), 4),
            "std": round(float(np.std(raw_spectra)), 4),
        }

        stats_proc = {
            "min": round(float(np.min(processed_spectra)), 4),
            "max": round(float(np.max(processed_spectra)), 4),
            "mean": round(float(np.mean(processed_spectra)), 4),
            "std": round(float(np.std(processed_spectra)), 4),
        }

        overall_status = "PASSED" if (nan_proc == 0 and inf_proc == 0 and labels_aligned) else "FAILED"

        report = {
            "overall_status": overall_status,
            "sample_counts": {
                "raw_samples": n_raw_samples,
                "processed_samples": n_proc_samples,
                "samples_preserved": bool(n_raw_samples == n_proc_samples),
            },
            "spectral_band_counts": {
                "raw_bands": n_raw_bands,
                "processed_bands": n_proc_bands,
            },
            "label_integrity": {
                "labels_aligned": labels_aligned,
            },
            "numerical_stability": {
                "nan_count": nan_proc,
                "inf_count": inf_proc,
                "is_stable": bool(nan_proc == 0 and inf_proc == 0),
            },
            "spectral_fidelity": fidelity,
            "distribution_comparison": {
                "raw_statistics": stats_raw,
                "processed_statistics": stats_proc,
            },
        }

        logger.info(f"Quality Control Audit Complete: Status={overall_status}, Fidelity={fidelity}")
        return report
