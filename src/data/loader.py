"""
Data Loader Module for Hyperspectral Datasets.
Supports loading tabular spectral CSVs, MATLAB .mat cubes, and ENVI/TIFF formats.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


def load_spectral_data(file_path: Union[str, Path]) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads spectral data from supported file formats (.csv, .mat, .npy).

    Parameters:
        file_path (Union[str, Path]): Path to data file.

    Returns:
        Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]: Loaded spectral dataframe or (cube, labels).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".npy":
        return np.load(path)
    elif suffix == ".mat":
        import scipy.io as sio
        mat = sio.loadmat(str(path))
        return mat
    else:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: .csv, .npy, .mat")
