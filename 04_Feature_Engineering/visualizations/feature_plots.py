"""
04_Feature_Engineering/visualizations/feature_plots.py

Publication-grade scientific visualizations for engineered features:
- Feature importance rankings (Random Forest with standard deviation bars)
- Engineered feature cross-correlation heatmaps
- Spectral derivative curves and Red-Edge Position (REP) shifts
- PCA Scree plot & multi-partition score distributions
- Continuum-relative water absorption depth comparisons
- Feature selection stage retention funnel
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.feature_utils import save_figure_safely


def plot_feature_importance_ranking(importance_df: pd.DataFrame, output_path: Path, top_k: int = 25, dpi: int = 300) -> None:
    """Plots horizontal bar chart of top K feature importances with standard deviation error bars."""
    top_df = importance_df.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_df))

    ax.barh(
        y_pos,
        top_df["importance"].values,
        xerr=top_df["importance_std"].values if "importance_std" in top_df.columns else None,
        color="teal",
        edgecolor="black",
        alpha=0.85,
        capsize=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"].values, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
    ax.set_title(f"TerraSpectra: Top {top_k} Most Predictive Features (Random Forest on Train Split)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_extracted_feature_correlation_heatmap(feature_df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plots Pearson correlation heatmap for engineered features."""
    corr = feature_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson Correlation ($r$)"},
        ax=ax,
    )
    ax.set_title("Engineered Feature Inter-Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_spectral_derivatives_profile(d1_matrix: np.ndarray, wavelengths: np.ndarray, disease_labels: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """Plots mean 1st derivative (dR/dlambda) spectra comparing Control and Infected."""
    ctrl_mask = (disease_labels == "Control")
    inf_mask = (disease_labels == "Infected")

    ctrl_d1_mean = np.mean(d1_matrix[ctrl_mask], axis=0)
    inf_d1_mean = np.mean(d1_matrix[inf_mask], axis=0)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(wavelengths, ctrl_d1_mean, color="#2e7d32", lw=2.0, label="Control Canopies ($dR/d\lambda$)")
    ax.plot(wavelengths, inf_d1_mean, color="#c62828", lw=2.0, label="Infected Canopies ($dR/d\lambda$)")
    ax.axvspan(680, 750, color="gold", alpha=0.22, label="Red-Edge Derivative Zone (680–750 nm)")
    ax.axhline(0, color="black", linestyle="--", lw=0.8)

    ax.set_title("First Spectral Derivative ($dR/d\lambda$) Curves: Healthy vs. Crown Rot", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel(r"First Derivative ($dR/d\lambda$)", fontsize=11)
    ax.set_xlim(450, 1100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_scree_and_variance(pca_model, output_path: Path, dpi: int = 300) -> None:
    """Plots PCA scree plot and cumulative explained variance curve."""
    ind_var = pca_model.explained_variance_ratio_ * 100
    cum_var = np.cumsum(ind_var)
    pcs = [f"PC{i+1}" for i in range(len(ind_var))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(pcs, ind_var, color="steelblue", edgecolor="black", alpha=0.85)
    ax1.plot(pcs, ind_var, color="darkblue", marker="o")
    ax1.set_title("PCA Scree Plot (Training Partition)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xlabel("Principal Component")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.plot(pcs, cum_var, color="forestgreen", marker="s", lw=2.0)
    ax2.axhline(95, color="red", linestyle="--", lw=1.2, label="95% Threshold")
    ax2.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Dimensionality Reduction: Training-Fitted PCA Explained Variance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_feature_selection_stages(stage_counts: Dict[str, int], output_path: Path, dpi: int = 300) -> None:
    """Plots retention funnel chart across the feature selection cascade."""
    stages = list(stage_counts.keys())
    counts = list(stage_counts.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, counts, color=["#4575b4", "#74add1", "#abd9e9", "#fdae61"], edgecolor="black", width=0.5)

    ax.set_title("Feature Selection Cascade: Feature Retention by Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Features Retained", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    plt.tight_layout()
    save_figure_safely(fig, output_path, dpi=dpi)
    plt.close(fig)
