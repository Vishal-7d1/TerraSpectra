"""
02_Data_Preprocessing/loaders/data_loader.py

Provides robust, format-aware loading of hyperspectral datasets.
Separates and validates metadata columns, sample identifiers, target disease labels,
and continuous spectral band reflectance matrices.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from utils.preprocessing_utils import setup_logger

logger = setup_logger("data_loader")


class HyperspectralDataLoader:
    """
    Handles ingestion of agricultural hyperspectral tabular matrices and image formats.
    """

    def __init__(self, data_path: Path, sheet_name: str = "barcode_hypdata"):
        self.data_path = Path(data_path)
        self.sheet_name = sheet_name

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """
        Loads the dataset and separates metadata from spectral matrix.

        Returns:
            Tuple:
                - metadata_df (pd.DataFrame): Non-spectral metadata and labels.
                - spectral_matrix (np.ndarray): Shape (n_samples, n_bands) float matrix.
                - wavelengths (np.ndarray): 1D array of wavelengths in nanometers.
                - spectral_col_names (List[str]): Original column headers for spectral bands.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Input hyperspectral dataset not found: {self.data_path}")

        ext = self.data_path.suffix.lower()
        logger.info(f"Loading hyperspectral dataset from: {self.data_path.name} (Format: {ext})...")

        if ext == ".xlsx":
            df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        elif ext == ".csv":
            df = pd.read_csv(self.data_path)
        elif ext == ".parquet":
            df = pd.read_parquet(self.data_path)
        elif ext in [".npy", ".npz"]:
            raw_arr = np.load(self.data_path)
            if isinstance(raw_arr, np.lib.npyio.NpzFile):
                return pd.DataFrame(), raw_arr["spectra"], raw_arr.get("wavelengths", np.arange(raw_arr["spectra"].shape[1])), []
            return pd.DataFrame(), raw_arr, np.arange(raw_arr.shape[1]), []
        else:
            raise ValueError(f"Unsupported format '{ext}'. Supported: .xlsx, .csv, .parquet, .npy, .npz")

        # Separate metadata columns vs spectral band columns
        spectral_cols = []
        metadata_cols = []
        wavelengths_list = []

        for col in df.columns:
            if isinstance(col, (int, float)):
                spectral_cols.append(col)
                wavelengths_list.append(float(col))
            elif isinstance(col, str) and (col.replace(".", "", 1).isdigit()):
                spectral_cols.append(col)
                wavelengths_list.append(float(col))
            elif isinstance(col, str) and (col.startswith("band_") and col.replace("band_", "").replace("nm", "").replace(".", "", 1).isdigit()):
                spectral_cols.append(col)
                wavelengths_list.append(float(col.replace("band_", "").replace("nm", "")))
            else:
                metadata_cols.append(col)

        if not spectral_cols:
            raise ValueError("No numeric spectral wavelength columns could be identified in dataset.")

        # Sort spectral bands monotonically by wavelength if not already sorted
        sorted_indices = np.argsort(wavelengths_list)
        spectral_cols = [spectral_cols[i] for i in sorted_indices]
        wavelengths = np.array([wavelengths_list[i] for i in sorted_indices], dtype=np.float64)

        metadata_df = df[metadata_cols].copy()
        spectral_matrix = df[spectral_cols].values.astype(np.float64)

        logger.info(
            f"Successfully loaded {len(df)} samples across {len(wavelengths)} spectral bands "
            f"({wavelengths.min():.2f} nm to {wavelengths.max():.2f} nm)."
        )

        return metadata_df, spectral_matrix, wavelengths, [str(c) for c in spectral_cols]
