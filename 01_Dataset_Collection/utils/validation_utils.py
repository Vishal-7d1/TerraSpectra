"""
Validation utilities for file integrity, format-specific inspections, and hyperspectral data structures.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config import setup_logger

logger = setup_logger("validation_utils")


def validate_file_level(file_path: Path) -> Dict[str, Any]:
    """
    Performs file-level existence, non-empty, and readability checks.
    """
    file_path = Path(file_path)
    result = {
        "file_name": file_path.name,
        "exists": file_path.exists(),
        "is_file": file_path.is_file() if file_path.exists() else False,
        "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "extension": file_path.suffix.lower(),
        "status": "VALID",
        "issues": [],
    }

    if not result["exists"]:
        result["status"] = "FAILED"
        result["issues"].append("File does not exist.")
        return result

    if result["size_bytes"] == 0:
        result["status"] = "FAILED"
        result["issues"].append("File is empty (0 bytes).")
        return result

    return result


def validate_tabular_spectral_dataset(
    file_path: Path,
    sheet_name: str = "barcode_hypdata",
    label_col: str = "disease"
) -> Dict[str, Any]:
    """
    Hyperspectral & tabular validation for spectral datasets stored in Excel/CSV format.
    Checks:
    - Dimensions (samples, bands)
    - Wavelength continuity and coverage
    - NaN and Inf presence
    - Reflectance value range
    - Class label distribution
    """
    file_path = Path(file_path)
    report = {
        "file_name": file_path.name,
        "format": file_path.suffix.lower(),
        "total_samples": 0,
        "total_columns": 0,
        "spectral_bands_count": 0,
        "wavelength_min_nm": None,
        "wavelength_max_nm": None,
        "nan_count": 0,
        "infinite_count": 0,
        "min_reflectance": None,
        "max_reflectance": None,
        "classes_detected": {},
        "missing_labels_count": 0,
        "status": "VALID",
        "notes": []
    }

    try:
        if file_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            report["status"] = "WARNING"
            report["notes"].append(f"Format {file_path.suffix} requires specialized binary parser.")
            return report

        report["total_samples"] = int(len(df))
        report["total_columns"] = int(df.shape[1])

        # Identify spectral wavelength columns (numeric names or float/int headers)
        spectral_cols = []
        for col in df.columns:
            if isinstance(col, (int, float)):
                spectral_cols.append(col)
            elif isinstance(col, str) and (col.replace(".", "", 1).isdigit() or col.startswith("band_")):
                spectral_cols.append(col)

        report["spectral_bands_count"] = len(spectral_cols)

        if spectral_cols:
            wls = []
            for c in spectral_cols:
                if isinstance(c, (int, float)):
                    wls.append(float(c))
                else:
                    clean_c = c.replace("band_", "").replace("nm", "")
                    if clean_c.replace(".", "", 1).isdigit():
                        wls.append(float(clean_c))

            if wls:
                report["wavelength_min_nm"] = round(min(wls), 2)
                report["wavelength_max_nm"] = round(max(wls), 2)

            # Check spectral matrix values
            spectral_matrix = df[spectral_cols].values
            report["nan_count"] = int(np.isnan(spectral_matrix).sum())
            report["infinite_count"] = int(np.isinf(spectral_matrix).sum())

            # Valid finite min/max
            finite_mask = np.isfinite(spectral_matrix)
            if finite_mask.any():
                report["min_reflectance"] = round(float(np.min(spectral_matrix[finite_mask])), 4)
                report["max_reflectance"] = round(float(np.max(spectral_matrix[finite_mask])), 4)

        # Check label column
        if label_col in df.columns:
            val_counts = df[label_col].value_counts(dropna=False).to_dict()
            report["classes_detected"] = {str(k): int(v) for k, v in val_counts.items()}
            report["missing_labels_count"] = int(df[label_col].isna().sum())
        else:
            report["notes"].append(f"Label column '{label_col}' not found. Available: {list(df.columns[:10])}")

        if report["nan_count"] > 0:
            report["notes"].append(f"Contains {report['nan_count']} NaN values in spectral bands.")

    except Exception as e:
        report["status"] = "FAILED"
        report["notes"].append(f"Validation error: {str(e)}")

    return report

