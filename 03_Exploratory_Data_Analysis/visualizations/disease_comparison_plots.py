"""
03_Exploratory_Data_Analysis/visualizations/disease_comparison_plots.py

Comparative scientific visualizations between Healthy (Control) and Crown Rot (Infected) canopies:
- Overlaid mean spectral signatures with standard deviation envelopes.
- Absolute Delta R and Relative Delta R_rel difference spectra.
- Manhattan-style statistical significance (-log10 p-value) with Benjamini-Hochberg FDR threshold.
- Cohen's d effect size spectrum across all 653 narrow bands.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.eda_utils import save_figure_safely


def plot_healthy_vs_diseased_signatures(diff_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots overlaid mean spectral curves with +/- 1 std confidence intervals for Control and Infected."""
    wavelengths = diff_df["wavelength_nm"].values
    ctrl_mean = diff_df["control_mean"].values
    ctrl_std = diff_df["control_std"].values
    inf_mean = diff_df["infected_mean"].values
    inf_std = diff_df["infected_std"].values

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Control
    ax.plot(wavelengths, ctrl_mean, color="#2e7d32", lw=2.2, label="Control / Healthy Canopies (N=816)")
    ax.fill_between(wavelengths, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color="#a5d6a7", alpha=0.3)

    # Infected
    ax.plot(wavelengths, inf_mean, color="#c62828", lw=2.2, label="Infected / Crown Rot Canopies (N=816)")
    ax.fill_between(wavelengths, inf_mean - inf_std, inf_mean + inf_std, color="#ef9a9a", alpha=0.3)

    # Red-Edge annotation window
    ax.axvspan(680, 750, color="gold", alpha=0.22, label="Red-Edge Boundary Transition (680–750 nm)")

    ax.set_title("Spectral Reflectance Profiles: Healthy (Control) vs. Crown Rot Infected Wheat", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance ($R$)", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_difference_spectra(diff_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots Absolute Difference (Delta R) and Percentage Relative Difference (Delta R_rel)."""
    wavelengths = diff_df["wavelength_nm"].values
    abs_diff = diff_df["absolute_difference"].values
    rel_diff = diff_df["relative_difference_pct"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Absolute Difference
    ax1.plot(wavelengths, abs_diff, color="#1565c0", lw=2.0, label=r"Absolute Difference $\Delta R = R_{inf} - R_{ctrl}$")
    ax1.axhline(0, color="black", linestyle="--", lw=1.0)
    ax1.axvspan(680, 750, color="gold", alpha=0.18, label="Red-Edge Region")
    ax1.set_ylabel(r"Absolute $\Delta$ Reflectance", fontsize=11)
    ax1.set_title("Difference Spectra (Infected minus Control)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9.5)

    # 2. Relative Difference (%)
    ax2.plot(wavelengths, rel_diff, color="#ad1457", lw=2.0, label=r"Relative Difference $\Delta R_{rel} = (\Delta R / R_{ctrl}) \times 100\%$")
    ax2.axhline(0, color="black", linestyle="--", lw=1.0)
    ax2.axvspan(680, 750, color="gold", alpha=0.18)
    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("Relative Change (%)", fontsize=11)
    ax2.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_wavelength_significance_manhattan(sig_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots -log10(p-value) across all wavelengths with FDR Benjamini-Hochberg threshold line."""
    wavelengths = sig_df["wavelength_nm"].values
    p_vals = np.clip(sig_df["welch_p_value_fdr_adjusted"].values, 1e-150, 1.0)
    neg_log10_p = -np.log10(p_vals)

    threshold_y = -np.log10(0.05)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(wavelengths, neg_log10_p, c=neg_log10_p, cmap="viridis", s=14, alpha=0.85, edgecolors="none")
    ax.axhline(threshold_y, color="red", linestyle="--", lw=1.5, label=rf"FDR Significance Threshold ($q = 0.05$, $-\log_{{10}}(q) = {threshold_y:.2f}$)")

    ax.set_title("Wavelength-Wise Statistical Significance (Welch's t-test with FDR Correction)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}(\text{FDR } q\text{-value})$", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_cohens_d_effect_size_spectrum(sig_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots Cohen's d effect size across the full spectrum highlighting large/medium effect thresholds."""
    wavelengths = sig_df["wavelength_nm"].values
    cohens_d = sig_df["cohens_d_effect_size"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavelengths, cohens_d, color="#4a148c", lw=1.8, label="Standardized Effect Size (Cohen's $d$)")
    ax.axhline(0, color="black", linestyle="--", lw=0.8)

    # Conventional Cohen's benchmarks
    ax.axhline(0.8, color="crimson", linestyle=":", lw=1.2, label=r"Large Positive Effect ($d \geq 0.8$)")
    ax.axhline(-0.8, color="crimson", linestyle=":", lw=1.2, label=r"Large Negative Effect ($d \leq -0.8$)")
    ax.axhline(0.5, color="orange", linestyle=":", lw=1.0, alpha=0.7, label=r"Medium Effect ($|d| \geq 0.5$)")
    ax.axhline(-0.5, color="orange", linestyle=":", lw=1.0, alpha=0.7)

    ax.set_title("Cohen's d Standardized Effect Size Spectrum (Crown Rot Infected vs. Control)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Effect Size ($d$)", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)
