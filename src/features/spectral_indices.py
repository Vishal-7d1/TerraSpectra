"""
Narrow-Band Spectral Vegetation Indices for Crop Disease & Stress Forecasting.

Hyperspectral bands (typical wavelengths in nm):
- Blue: ~450 nm
- Green: ~550 nm
- Red: ~670 nm
- Red Edge: ~705 - 720 nm
- NIR (Near-Infrared): ~800 nm
- Xanthophyll reference: ~531 nm & 570 nm
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Safe element-wise division avoiding divide-by-zero."""
    denom = np.where(denominator == 0, eps, denominator)
    return numerator / denom


def calculate_ndvi(r_nir: np.ndarray, r_red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI)
    NDVI = (NIR - Red) / (NIR + Red)
    General canopy vigor, biomass, and chlorophyll absorption.
    """
    return safe_divide(r_nir - r_red, r_nir + r_red)


def calculate_ndre(r_nir: np.ndarray, r_red_edge: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Red Edge (NDRE)
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    Detects subtle chlorophyll changes and early pre-symptomatic disease stress.
    """
    return safe_divide(r_nir - r_red_edge, r_nir + r_red_edge)


def calculate_gndvi(r_nir: np.ndarray, r_green: np.ndarray) -> np.ndarray:
    """
    Green Normalized Difference Vegetation Index (GNDVI)
    GNDVI = (NIR - Green) / (NIR + Green)
    Sensitive to wide range of chlorophyll content.
    """
    return safe_divide(r_nir - r_green, r_nir + r_green)


def calculate_pri(r_531: np.ndarray, r_570: np.ndarray) -> np.ndarray:
    """
    Photochemical Reflectance Index (PRI)
    PRI = (R531 - R570) / (R531 + R570)
    Measures xanthophyll cycle activity; highly sensitive to physiological stress.
    """
    return safe_divide(r_531 - r_570, r_531 + r_570)


def calculate_psri(r_red: np.ndarray, r_blue: np.ndarray, r_nir: np.ndarray) -> np.ndarray:
    """
    Plant Senescence Reflectance Index (PSRI)
    PSRI = (R680 - R500) / R750
    Increases during senescence, tissue necrosis, and fungal pathogen invasion.
    """
    return safe_divide(r_red - r_blue, r_nir)


def extract_vegetation_indices(band_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Extracts a dictionary/dataframe of key narrow-band vegetation indices.

    Expects band_dict with keys: 'blue', 'green', 'red', 'red_edge', 'nir', etc.
    """
    features = {}

    if "nir" in band_dict and "red" in band_dict:
        features["NDVI"] = calculate_ndvi(band_dict["nir"], band_dict["red"])

    if "nir" in band_dict and "red_edge" in band_dict:
        features["NDRE"] = calculate_ndre(band_dict["nir"], band_dict["red_edge"])

    if "nir" in band_dict and "green" in band_dict:
        features["GNDVI"] = calculate_gndvi(band_dict["nir"], band_dict["green"])

    if "r531" in band_dict and "r570" in band_dict:
        features["PRI"] = calculate_pri(band_dict["r531"], band_dict["r570"])

    if "red" in band_dict and "blue" in band_dict and "nir" in band_dict:
        features["PSRI"] = calculate_psri(band_dict["red"], band_dict["blue"], band_dict["nir"])

    return pd.DataFrame(features)

