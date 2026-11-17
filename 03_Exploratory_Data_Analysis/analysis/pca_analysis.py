"""
03_Exploratory_Data_Analysis/analysis/pca_analysis.py

Unsupervised Principal Component Analysis (PCA) for hyperspectral dimensionality reduction:
- Evaluates intrinsic dimensionality of the 653 narrow spectral bands.
- Computes explained variance ratios and cumulative variance.
- Extracts band-wise component loadings to map principal axes back to physical wavelengths.
- Generates score coordinates strictly without utilizing target disease labels as input features.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.eda_utils import save_json


class SpectralPCAAnalyzer:
    """Performs rigorous unsupervised PCA on high-dimensional hyperspectral signatures."""

    def __init__(self, metadata_df: pd.DataFrame, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.metadata_df = metadata_df
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def fit_pca(self, n_components: int = 5, random_state: int = 42) -> Tuple[PCA, np.ndarray, np.ndarray]:
        """
        Standardizes spectral features (zero mean, unit variance) and fits PCA.
        Returns:
            pca: Fitted sklearn PCA model
            scores: Projection of observations in PC space (N_samples, n_components)
            standardized_matrix: Scaled spectral array
        """
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(self.spectral_matrix)

        pca = PCA(n_components=n_components, random_state=random_state)
        scores = pca.fit_transform(scaled_matrix)

        return pca, scores, scaled_matrix

    def run_pca_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Executes PCA, exports explained variance, loadings, and scores."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_comp = config.get("pca", {}).get("n_components", 5)
        seed = config.get("pca", {}).get("random_state", 42)

        pca, scores, _ = self.fit_pca(n_components=n_comp, random_state=seed)

        # 1. Explained Variance Table
        var_ratios = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratios)

        var_df = pd.DataFrame({
            "principal_component": [f"PC{i+1}" for i in range(n_comp)],
            "explained_variance_ratio": var_ratios,
            "explained_variance_percentage": np.round(var_ratios * 100, 2),
            "cumulative_variance_ratio": cum_var,
            "cumulative_variance_percentage": np.round(cum_var * 100, 2),
        })
        var_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

        # 2. Component Loadings across Wavelengths
        loadings_dict = {"wavelength_nm": self.wavelengths}
        for i in range(n_comp):
            loadings_dict[f"PC{i+1}_loading"] = pca.components_[i]
        loadings_df = pd.DataFrame(loadings_dict)
        loadings_df.to_csv(output_dir / "pca_loadings.csv", index=False)

        # 3. Sample Scores with Metadata Alignment
        scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])
        # Merge with select metadata for visualization without data leakage
        meta_subset = self.metadata_df.copy()
        combined_scores_df = pd.concat([meta_subset.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
        combined_scores_df.to_csv(output_dir / "pca_scores.csv", index=False)

        # 4. Dominant Physical Bands for PC1 and PC2
        top_pc1_band_idx = int(np.argmax(np.abs(pca.components_[0])))
        top_pc2_band_idx = int(np.argmax(np.abs(pca.components_[1])))

        summary = {
            "n_components_fitted": int(n_comp),
            "pc1_explained_variance_pct": round(float(var_ratios[0] * 100), 2),
            "pc2_explained_variance_pct": round(float(var_ratios[1] * 100), 2),
            "cumulative_variance_top2_pct": round(float(cum_var[1] * 100), 2),
            "total_explained_variance_pct": round(float(cum_var[-1] * 100), 2),
            "pc1_dominant_wavelength_nm": float(self.wavelengths[top_pc1_band_idx]),
            "pc2_dominant_wavelength_nm": float(self.wavelengths[top_pc2_band_idx]),
            "interpretation_note": "PC1 captures general broadband albedo/brightness scattering variations across the canopy; PC2 captures red-edge chlorophyll transitions and water absorption inflections."
        }
        save_json(summary, output_dir / "pca_summary.json")

        return summary
