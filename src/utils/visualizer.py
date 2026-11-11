"""
Visualization utilities for Hyperspectral Curves and Model Evaluation.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_spectral_curves(
    wavelengths: np.ndarray,
    healthy_spectra: np.ndarray,
    diseased_spectra: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plots mean spectral reflectance curves comparing healthy vs infected crops.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, np.mean(healthy_spectra, axis=0), label="Healthy Crop", color="green", lw=2)
    plt.plot(wavelengths, np.mean(diseased_spectra, axis=0), label="Diseased Crop", color="crimson", lw=2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Mean Hyperspectral Reflectance: Healthy vs. Diseased")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a heatmap confusion matrix.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Plots top-K most influential spectral bands or vegetation indices.
    """
    indices = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(8, 6))
    plt.barh(range(top_k), importances[indices][::-1], align="center", color="teal")
    plt.yticks(range(top_k), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Relative Importance")
    plt.title(f"Top {top_k} Discriminative Spectral Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

