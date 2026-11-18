"""
03_Exploratory_Data_Analysis/utils/eda_utils.py

Shared scientific and operational utility functions for the TerraSpectra EDA pipeline.
Includes dataset loading, wavelength mapping, config parsing, safe file saving,
and statistical effect size functions.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).resolve().parent.parent


def setup_logger(name: str = "eda_pipeline", log_file: Optional[Path] = None) -> logging.Logger:
    """Configures a standardized logger with dual stream and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_dir = MODULE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "eda.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and returns the EDA pipeline configuration dictionary."""
    if config_path is None:
        config_path = MODULE_DIR / "configs" / "eda_config.json"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_dataset(filepath: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Loads hyperspectral dataset from Parquet or CSV.
    Separates metadata from continuous spectral wavelength columns.
    Returns:
        metadata_df: DataFrame containing all non-spectral attributes
        spectral_matrix: 2D numpy array of shape (N_samples, N_bands)
        wavelengths: 1D numpy array of wavelength values in nm (float)
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
        # Check if column represents a numeric wavelength
        try:
            wl_val = float(col_str)
            spectral_cols.append(col)
            wavelengths_list.append(wl_val)
        except ValueError:
            metadata_cols.append(col)

    if not spectral_cols:
        raise ValueError("No numeric spectral wavelength columns could be identified in dataset.")

    # Sort spectral columns strictly by ascending wavelength
    sorted_pairs = sorted(zip(wavelengths_list, spectral_cols), key=lambda x: x[0])
    wavelengths = np.array([p[0] for p in sorted_pairs], dtype=np.float64)
    spectral_cols = [p[1] for p in sorted_pairs]

    metadata_df = df[metadata_cols].copy()
    spectral_matrix = df[spectral_cols].to_numpy(dtype=np.float64)

    return metadata_df, spectral_matrix, wavelengths, spectral_cols


def find_nearest_wavelength(target_nm: float, available_wls: np.ndarray) -> Tuple[int, float]:
    """
    Finds the index and exact value of the nearest available wavelength in the dataset.
    """
    idx = int(np.argmin(np.abs(available_wls - target_nm)))
    actual_nm = float(available_wls[idx])
    return idx, actual_nm


def calculate_auc(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes Area Under Curve using trapezoidal rule, fully compatible with NumPy 2.x and SciPy."""
    from scipy.integrate import trapezoid
    return trapezoid(y, x, axis=axis)


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cohen's d effect size between two independent groups:
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def save_figure_safely(fig, path: Path, dpi: int = 300) -> None:
    """
    Saves a matplotlib figure safely with retry logic to avoid OneDrive / Windows file sync locks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            return
        except OSError:
            time.sleep(0.5)
    # Final attempt fallback
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_json(data: Any, path: Path, indent: int = 4) -> None:
    """Saves a Python dict or list to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
