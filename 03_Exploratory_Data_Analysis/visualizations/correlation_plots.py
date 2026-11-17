"""
03_Exploratory_Data_Analysis/visualizations/correlation_plots.py

Scientific correlation and multivariate PCA visualizations:
- Subsampled cross-band Pearson correlation matrix heatmap.
- Vegetation indices association bar chart (Point-Biserial r).
- PCA Scree plot & Cumulative explained variance curve.
- 2D PCA score scatter plot colored by disease status.
- Principal component spectral loadings across all 653 wavelengths.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.eda_utils import save_figure_safely


def plot_band_correlation_heatmap(corr_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots cross-band Pearson correlation heatmap for subsampled wavelength bands."""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Downsample labels for readability if more than 30 bands
    step = max(1, len(corr_df) // 15)
    xtick_labels = [corr_df.columns[i] if i % step == 0 else "" for i in range(len(corr_df.columns))]
    ytick_labels = [corr_df.index[i] if i % step == 0 else "" for i in range(len(corr_df.index))]

    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        cbar_kws={"label": "Pearson Correlation ($r$)"},
        ax=ax,
    )

    ax.set_title("Hyperspectral Cross-Band Correlation Heatmap (Inter-Wavelength Collinearity)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_indices_disease_correlation(ind_corr_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots horizontal bar chart of Point-Biserial correlation between vegetation indices and Crown Rot infection."""
    fig, ax = plt.subplots(figsize=(9, 5))

    indices = ind_corr_df["vegetation_index"].values
    r_vals = ind_corr_df["point_biserial_r"].values
    colors = ["#c62828" if r > 0 else "#1565c0" for r in r_vals]

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, r_vals, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", lw=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(indices, fontsize=10, fontweight="bold")
    ax.set_xlabel("Point-Biserial Correlation ($r$) with Disease Infection (Control=0, Infected=1)", fontsize=10)
    ax.set_title("Vegetation Index Correlation with Crown Rot Infection", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Value labels on bars
    for i, val in enumerate(r_vals):
        offset = 0.02 if val >= 0 else -0.06
        ax.text(val + offset, i, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_scree_and_variance(var_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots PCA scree plot (individual variance) and cumulative explained variance curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    pcs = var_df["principal_component"].values
    ind_pct = var_df["explained_variance_percentage"].values
    cum_pct = var_df["cumulative_variance_percentage"].values

    # Scree plot
    ax1.bar(pcs, ind_pct, color="steelblue", edgecolor="black", alpha=0.85)
    ax1.plot(pcs, ind_pct, color="darkblue", marker="o", lw=1.8)
    ax1.set_title("PCA Scree Plot (Individual Explained Variance)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xlabel("Principal Component")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i, pct in enumerate(ind_pct):
        ax1.text(i, pct + 0.8, f"{pct:.1f}%", ha="center", fontsize=9)

    # Cumulative variance plot
    ax2.plot(pcs, cum_pct, color="forestgreen", marker="s", lw=2.2, label="Cumulative Variance")
    ax2.axhline(95, color="red", linestyle="--", lw=1.2, label="95% Variance Threshold")
    ax2.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    for i, pct in enumerate(cum_pct):
        ax2.text(i, pct - 4.5, f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Unsupervised Hyperspectral Dimensionality Reduction (PCA)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_scores_disease_scatter(scores_df: pd.DataFrame, disease_col: str, output_path: Path, dpi: int = 300) -> None:
    """Plots 2D scatter of sample scores on PC1 vs PC2 colored by Disease label."""
    fig, ax = plt.subplots(figsize=(9, 7))

    palette = {"Control": "#2ca02c", "Infected": "#d62728"}

    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue=disease_col,
        data=scores_df,
        palette=palette,
        alpha=0.65,
        s=40,
        edgecolor="none",
        ax=ax,
    )

    ax.set_title("PCA Projection: PC1 vs. PC2 Colored by Crop Health Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Disease Class", loc="best", framealpha=0.9)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_component_loadings(loadings_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots spectral loading curves for PC1, PC2, and PC3 across the 653 wavelengths."""
    wavelengths = loadings_df["wavelength_nm"].values

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(wavelengths, loadings_df["PC1_loading"], color="#1f77b4", lw=1.8, label="PC1 Loadings (Albedo / Canopy Bulk Scatter)")
    ax.plot(wavelengths, loadings_df["PC2_loading"], color="#ff7f0e", lw=1.8, label="PC2 Loadings (Red-Edge / Chlorophyll / Water)")
    if "PC3_loading" in loadings_df.columns:
        ax.plot(wavelengths, loadings_df["PC3_loading"], color="#2ca02c", lw=1.5, alpha=0.8, label="PC3 Loadings")

    ax.axhline(0, color="black", linestyle="--", lw=0.8)
    ax.axvspan(680, 750, color="gold", alpha=0.18, label="Red-Edge Region")

    ax.set_title("Principal Component Spectral Loadings Across 653 Narrow Bands", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Eigenvector Loading Weight", fontsize=11)
    ax.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)
