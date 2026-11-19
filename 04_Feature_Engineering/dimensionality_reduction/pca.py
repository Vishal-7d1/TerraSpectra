"""
04_Feature_Engineering/dimensionality_reduction/pca.py

Unsupervised Principal Component Analysis (PCA) for dimensionality reduction:
- Fits StandardScaler and PCA strictly on the training partition.
- Applies fitted transformation to validation and test partitions without leakage.
- Serializes fitted transformer to joblib for production inference.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from utils.feature_utils import save_json


class LeakageFreePCA:
    """Standardized PCA transformer with strict fit/transform separation."""

    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False
        self.feature_names_in_: Optional[list] = None

    def fit(self, X_train: pd.DataFrame) -> "LeakageFreePCA":
        """Fits scaler and PCA strictly on training data."""
        self.feature_names_in_ = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms any partition using training-fitted scaler and PCA."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted on training data before transforming.")
        X_scaled = self.scaler.transform(X[self.feature_names_in_])
        scores = self.pca.transform(X_scaled)
        cols = [f"PCA_component_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(scores, columns=cols, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms training data."""
        return self.fit(X_train).transform(X_train)

    def save_fitted_pipeline(self, output_path: Path) -> None:
        """Serializes the fitted scaler and PCA model into a single artifact."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted PCA pipeline.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "feature_names_in": self.feature_names_in_,
            "n_components": self.n_components,
        }, output_path)

    @classmethod
    def load_fitted_pipeline(cls, model_path: Path) -> "LeakageFreePCA":
        """Loads a serialized pipeline."""
        payload = joblib.load(model_path)
        instance = cls(n_components=payload["n_components"])
        instance.scaler = payload["scaler"]
        instance.pca = payload["pca"]
        instance.feature_names_in_ = payload["feature_names_in"]
        instance.is_fitted = True
        return instance
