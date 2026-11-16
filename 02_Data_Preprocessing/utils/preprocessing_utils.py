"""
02_Data_Preprocessing/utils/preprocessing_utils.py

Provides common utility functions for logging, configuration loading,
file serialization (CSV, Parquet, JSON, Markdown), and scientific plotting.
"""

from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = MODULE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "preprocessing.log"


def setup_logger(name: str = "preprocessing") -> logging.Logger:
    """Configures and returns a thread-safe logger writing to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads preprocessing configuration JSON from file."""
    if config_path is None:
        config_path = MODULE_DIR / "configs" / "preprocessing_config.json"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serializes data into formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, base_path_no_ext: Path, save_csv: bool = True, save_parquet: bool = True) -> Dict[str, str]:
    """Saves DataFrame as CSV and/or Parquet format."""
    saved_paths = {}
    base_path = Path(base_path_no_ext)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if save_csv:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        saved_paths["csv"] = str(csv_path)

    if save_parquet:
        try:
            parquet_path = base_path.with_suffix(".parquet")
            # Parquet requires all column names to be strings
            df_parquet = df.copy()
            df_parquet.columns = [str(c) for c in df_parquet.columns]
            df_parquet.to_parquet(parquet_path, index=False)
            saved_paths["parquet"] = str(parquet_path)
        except Exception as e:
            logging.getLogger("preprocessing_utils").warning(f"Parquet export skipped: {e}")

    return saved_paths


def save_figure_safely(fig, path: Path, dpi: int = 300) -> None:
    """Saves a matplotlib figure safely with retries against file locks (e.g. OneDrive)."""
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=dpi)
            return
        except OSError:
            time.sleep(0.6)
    fig.savefig(path, dpi=dpi)


def plot_spectral_comparison(
    wavelengths: np.ndarray,
    original_spectra: np.ndarray,
    processed_spectra: np.ndarray,
    title: str,
    original_label: str = "Raw",
    processed_label: str = "Processed",
    save_path: Optional[Path] = None,
    num_samples_to_plot: int = 5
) -> None:
    """
    Plots side-by-side or overlaid spectral curves before and after a transformation.
    """
    plt.figure(figsize=(10, 5))
    indices = np.linspace(0, len(original_spectra) - 1, num_samples_to_plot, dtype=int)

    for idx in indices:
        plt.plot(wavelengths, original_spectra[idx], color="gray", alpha=0.4, linestyle="--",
                 label=f"{original_label}" if idx == indices[0] else "")
        plt.plot(wavelengths, processed_spectra[idx], color="teal", alpha=0.8, lw=1.5,
                 label=f"{processed_label}" if idx == indices[0] else "")

    plt.xlabel("Wavelength (nm)", fontsize=11)
    plt.ylabel("Reflectance", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        save_figure_safely(plt.gcf(), save_path, dpi=300)
    plt.close()
