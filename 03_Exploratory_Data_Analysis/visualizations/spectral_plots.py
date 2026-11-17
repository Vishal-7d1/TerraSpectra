"""
03_Exploratory_Data_Analysis/visualizations/spectral_plots.py

Publication-grade hyperspectral curve visualizations:
- Global mean spectral signature with 1-sigma & 2-sigma confidence ribbons.
- Biophysical spectral region partitioning (VIS, Red-Edge, NIR, SWIR-1, SWIR-2).
- Wavelength-wise spectral variability and coefficient of variation (CV).
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.eda_utils import save_figure_safely


def plot_global_spectral_signature(spectral_matrix: np.ndarray, wavelengths: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """Plots global mean reflectance spectrum across all samples with standard deviation bands."""
    mean_curve = np.mean(spectral_matrix, axis=0)
    std_curve = np.std(spectral_matrix, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(wavelengths, mean_curve, color="#1b5e20", lw=2.2, label=f"Global Mean Reflectance (N={spectral_matrix.shape[0]:,})")
    ax.fill_between(wavelengths, mean_curve - std_curve, mean_curve + std_curve,
                    color="#81c784", alpha=0.35, label="$\pm 1\sigma$ Standard Deviation Envelope")
    ax.fill_between(wavelengths, mean_curve - 2 * std_curve, mean_curve + 2 * std_curve,
                    color="#c8e6c9", alpha=0.25, label="$\pm 2\sigma$ Range (95% Variation)")

    ax.set_title("TerraSpectra: Global Hyperspectral Reflectance Profile (450 nm – 2400 nm)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance ($R$)", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.set_ylim(0.0, float(np.max(mean_curve + 2 * std_curve) * 1.1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_spectral_regions_breakdown(spectral_matrix: np.ndarray, wavelengths: np.ndarray, config: Dict[str, Any], output_path: Path, dpi: int = 300) -> None:
    """Plots mean spectrum overlaid with physical and physiological spectral domains."""
    mean_curve = np.mean(spectral_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(wavelengths, mean_curve, color="black", lw=2.0, label="Canopy Mean Reflectance")

    # Define domain shading
    regions = [
        ("Visible (VIS: 450–700 nm)\nChlorophyll/Carotenoids", 450.0, 700.0, "#a8e6cf"),
        ("Red-Edge (680–750 nm)\nCellular Transition", 680.0, 750.0, "#ffd3b6"),
        ("Near-Infrared (NIR: 750–1300 nm)\nMesophyll Scattering Plateau", 750.0, 1300.0, "#dcedc1"),
        ("SWIR-1 (1300–1900 nm)\nMoisture & Protein Bands", 1300.0, 1900.0, "#ffaaa5"),
        ("SWIR-2 (1900–2400 nm)\nLignin & Cellulose", 1900.0, 2400.0, "#ff8b94"),
    ]

    for label, w_min, w_max, color in regions:
        ax.axvspan(w_min, w_max, color=color, alpha=0.28, label=label)

    # Indicate known atmospheric/liquid water absorption dips (~1450 nm and ~1940 nm)
    ax.axvline(1450, color="navy", linestyle=":", lw=1.5, alpha=0.7, label="Water Absorption Center (~1450 nm)")
    ax.axvline(1940, color="navy", linestyle=":", lw=1.5, alpha=0.7, label="Water Absorption Center (~1940 nm)")

    ax.set_title("Hyperspectral Domain Characterization Across Biophysical Regions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance ($R$)", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98), fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_spectral_variability(spectral_matrix: np.ndarray, wavelengths: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """Plots Coefficient of Variation (CV = std / mean) across all 653 wavelengths."""
    means = np.mean(spectral_matrix, axis=0)
    stds = np.std(spectral_matrix, axis=0, ddof=1)
    cv = np.where(means != 0, stds / means, 0.0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wavelengths, cv * 100, color="darkviolet", lw=1.8, label="Spectral Coefficient of Variation (CV %)")
    ax.set_title("Wavelength-Wise Spectral Variability Across the Agricultural Canopy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Coefficient of Variation (%)", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)
