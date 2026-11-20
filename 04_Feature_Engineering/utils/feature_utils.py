"""
04_Feature_Engineering/utils/feature_utils.py

Shared utilities for the TerraSpectra Feature Engineering module:
- Logging configuration
- Dataset ingestion and band separation
- Nearest wavelength matching
- Safe multi-format serialization (Parquet, CSV, Joblib, JSON)
- Thread-safe / lock-safe figure saving
- SHA-256 cryptographic verification
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

MODULE_DIR = Path(__file__).resolve().parent.parent


def setup_logger(name: str = "feature_engineering", log_file: Optional[Path] = None) -> logging.Logger:
    """Configures standardized logging to console and disk."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is None:
        log_dir = MODULE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "feature_engineering.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration JSON."""
    if config_path is None:
        config_path = MODULE_DIR / "configs" / "feature_config.json"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_dataset(filepath: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Loads hyperspectral dataset from Parquet or CSV.
    Separates metadata attributes from continuous spectral reflectance bands.
    Returns:
        metadata_df: DataFrame of metadata columns
        spectral_matrix: 2D numpy array (N_samples, N_bands)
        wavelengths: 1D numpy array of float wavelengths in nm
        spectral_cols: List of column names corresponding to spectral bands
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Hyperspectral dataset file not found: {filepath}")

    if filepath.suffix.lower() == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(filepath)
    elif filepath.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    spectral_cols = []
    wavelengths_list = []
    metadata_cols = []

    for col in df.columns:
        col_str = str(col).strip()
        try:
            wl_val = float(col_str)
            spectral_cols.append(col)
            wavelengths_list.append(wl_val)
        except ValueError:
            metadata_cols.append(col)

    if not spectral_cols:
        raise ValueError("No numeric spectral bands found in dataset.")

    # Sort spectral columns strictly by ascending wavelength
    sorted_pairs = sorted(zip(wavelengths_list, spectral_cols), key=lambda x: x[0])
    wavelengths = np.array([p[0] for p in sorted_pairs], dtype=np.float64)
    spectral_cols = [p[1] for p in sorted_pairs]

    metadata_df = df[metadata_cols].copy()
    spectral_matrix = df[spectral_cols].to_numpy(dtype=np.float64)

    return metadata_df, spectral_matrix, wavelengths, spectral_cols


def find_nearest_wavelength(target_nm: float, available_wls: np.ndarray) -> Tuple[int, float]:
    """Finds index and exact value of nearest available wavelength in nanometers."""
    idx = int(np.argmin(np.abs(available_wls - target_nm)))
    return idx, float(available_wls[idx])


def save_dataframe(
    df: pd.DataFrame,
    base_path_no_ext: Path,
    save_csv: bool = True,
    save_parquet: bool = True
) -> Dict[str, str]:
    """Exports DataFrame to Parquet and/or CSV."""
    saved = {}
    base_path = Path(base_path_no_ext)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if save_csv:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        saved["csv"] = str(csv_path)

    if save_parquet:
        try:
            parquet_path = base_path.with_suffix(".parquet")
            df_pq = df.copy()
            df_pq.columns = [str(c) for c in df_pq.columns]
            df_pq.to_parquet(parquet_path, index=False)
            saved["parquet"] = str(parquet_path)
        except Exception as e:
            logging.getLogger("feature_utils").warning(f"Parquet export failed: {e}")

    return saved


def save_figure_safely(fig, path: Path, dpi: int = 300) -> None:
    """Saves a matplotlib figure safely with retry logic against Windows file sync locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            return
        except OSError:
            time.sleep(0.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_json(data: Any, path: Path, indent: int = 4) -> None:
    """Saves a dictionary or list to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


calculate_sha256 = compute_sha256
