"""
03_Exploratory_Data_Analysis/visualizations/distribution_plots.py

Publication-quality distribution visualizations:
- Experimental factor balances (disease, watering, cultivar, species).
- Vegetation index distributions stratified by crop disease status.
- Spectral AUC outlier diagnostic distributions.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.eda_utils import save_figure_safely, calculate_auc


def plot_dataset_overview_dashboard(metadata_df: pd.DataFrame, config: Dict[str, Any], output_path: Path, dpi: int = 300) -> None:
    """Generates a 4-panel dashboard showing class and experimental treatment balances."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    disease_col = config["data_inputs"].get("target_disease_column", "disease")
    watering_col = config["data_inputs"].get("watering_column", "watering")
    cultivar_col = config["data_inputs"].get("cultivar_column", "genotype ID")
    species_col = config["data_inputs"].get("species_column", "Plant Species")

    # 1. Disease distribution
    if disease_col in metadata_df.columns:
        counts = metadata_df[disease_col].value_counts()
        colors = ["#2ca02c", "#d62728"] if "Control" in counts.index else sns.color_palette("Set2")
        axes[0, 0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=colors,
                       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        axes[0, 0].set_title("Disease Class Distribution\n(Control vs. Crown Rot Infected)", fontsize=11, fontweight="bold")

    # 2. Watering condition distribution
    if watering_col in metadata_df.columns:
        counts = metadata_df[watering_col].value_counts()
        colors = ["#1f77b4", "#ff7f0e"]
        axes[0, 1].bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
        axes[0, 1].set_title("Watering Treatment Distribution", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Number of Observations")
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Cultivar / Genotype distribution
    if cultivar_col in metadata_df.columns:
        counts = metadata_df[cultivar_col].value_counts()
        palette = sns.color_palette("muted", len(counts))
        axes[1, 0].bar(counts.index.astype(str), counts.values, color=palette, edgecolor="black", alpha=0.85)
        axes[1, 0].set_title("Wheat Cultivars / Genotypes Distribution", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Number of Observations")
        axes[1, 0].set_xlabel("Genotype ID")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Species distribution
    if species_col in metadata_df.columns:
        counts = metadata_df[species_col].value_counts()
        palette = sns.color_palette("pastel", len(counts))
        axes[1, 1].bar(counts.index.astype(str), counts.values, color=palette, edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("Plant Species Distribution\n(T. aestivum vs. T. durum)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Number of Observations")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("TerraSpectra: Experimental Dataset Profiling Dashboard", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_vegetation_indices_distributions(indices_df: pd.DataFrame, metadata_df: pd.DataFrame, config: Dict[str, Any], output_path: Path, dpi: int = 300) -> None:
    """Generates comparative boxplots for all calculated vegetation indices between Control and Infected."""
    disease_col = config["data_inputs"].get("target_disease_column", "disease")
    if disease_col not in metadata_df.columns:
        return

    cols = list(indices_df.columns)
    n_cols = len(cols)
    rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    combined_df = pd.concat([indices_df, metadata_df[[disease_col]]], axis=1)
    palette = {"Control": "#2ca02c", "Infected": "#d62728"}

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.boxplot(x=disease_col, y=col, data=combined_df, ax=ax, palette=palette, width=0.45,
                    boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markersize=3, alpha=0.3))
        sns.stripplot(x=disease_col, y=col, data=combined_df, ax=ax, color="black", alpha=0.15, jitter=0.2, size=2.5)
        ax.set_title(f"{col} Index Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Vegetation & Physiological Indices: Healthy (Control) vs. Crown Rot Infected", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_outlier_auc_distribution(spectral_matrix: np.ndarray, wavelengths: np.ndarray, bounds: Dict[str, float], output_path: Path, dpi: int = 300) -> None:
    """Plots distribution of spectral Area Under Curve (AUC) with 2.5x IQR outlier threshold boundaries."""
    auc_values = calculate_auc(spectral_matrix, wavelengths, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram + KDE
    sns.histplot(auc_values, kde=True, ax=ax1, color="steelblue", bins=35)
    ax1.axvline(bounds["lower_bound"], color="red", linestyle="--", lw=1.8, label=f"Lower Threshold ({bounds['lower_bound']:.1f})")
    ax1.axvline(bounds["upper_bound"], color="red", linestyle="--", lw=1.8, label=f"Upper Threshold ({bounds['upper_bound']:.1f})")
    ax1.axvline(bounds["q25"], color="gray", linestyle=":", label="Q1 (25th)")
    ax1.axvline(bounds["q75"], color="gray", linestyle=":", label="Q3 (75th)")
    ax1.set_title("Spectral AUC Distribution with IQR Outlier Bounds", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Integrated Spectral AUC ($\int R(\lambda) d\lambda$)")
    ax1.set_ylabel("Observation Count")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Boxplot
    sns.boxplot(x=auc_values, ax=ax2, color="lightblue", width=0.4)
    ax2.set_title("Spectral AUC Boxplot", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Integrated Spectral AUC")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Hyperspectral Anomaly & Outlier Screening (2.5x IQR Rule)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)
