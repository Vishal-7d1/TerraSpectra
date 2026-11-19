"""
04_Feature_Engineering/selection/variance_filter.py

Variance-based feature screening:
Removes constant and near-constant features below a configurable variance threshold.
Strictly fit only on the training partition to prevent data leakage.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class LeakageFreeVarianceFilter:
    """Filters low-variance features with strict fit/transform separation."""

    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
        self.retained_columns: Optional[List[str]] = None
        self.dropped_columns: Optional[List[str]] = None
        self.feature_variances_: Optional[Dict[str, float]] = None

    def fit(self, X_train: pd.DataFrame) -> "LeakageFreeVarianceFilter":
        """Fits variance threshold strictly on training data."""
        self.selector.fit(X_train)
        variances = X_train.var().to_dict()
        self.feature_variances_ = {k: float(v) for k, v in variances.items()}

        support = self.selector.get_support()
        self.retained_columns = list(X_train.columns[support])
        self.dropped_columns = list(X_train.columns[~support])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms any dataset using the columns selected during fit."""
        if self.retained_columns is None:
            raise RuntimeError("Filter must be fit on training data before transforming.")
        return X[self.retained_columns].copy()

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms training data."""
        return self.fit(X_train).transform(X_train)
