"""
04_Feature_Engineering/selection/correlation_filter.py

Multicollinearity reduction filter:
Identifies pairs of features with Pearson correlation |r| >= threshold on training data.
Retains the feature with higher target association (point-biserial / F-statistic)
and removes the collinear redundancy.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


class LeakageFreeCorrelationFilter:
    """Removes redundant collinear features with strict fit/transform separation."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.retained_columns: Optional[List[str]] = None
        self.dropped_columns: Optional[List[str]] = None
        self.correlated_pairs_: Optional[List[Dict[str, Any]]] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "LeakageFreeCorrelationFilter":
        """Identifies collinear clusters strictly on training observations."""
        corr_matrix = X_train.corr().abs()
        cols = list(X_train.columns)

        # Compute univariate association of each feature with binary target
        target_associations = {}
        for c in cols:
            r_val, _ = stats.pointbiserialr(y_train, X_train[c].values)
            target_associations[c] = abs(r_val) if not np.isnan(r_val) else 0.0

        dropped = set()
        pairs = []

        for i in range(len(cols)):
            col_i = cols[i]
            if col_i in dropped:
                continue
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in dropped:
                    continue
                r = float(corr_matrix.loc[col_i, col_j])
                if r >= self.threshold:
                    pairs.append({
                        "feature_1": col_i,
                        "feature_2": col_j,
                        "pearson_r": round(r, 4),
                        "score_1": round(target_associations[col_i], 4),
                        "score_2": round(target_associations[col_j], 4),
                    })
                    # Drop the one with lower target association
                    if target_associations[col_i] >= target_associations[col_j]:
                        dropped.add(col_j)
                    else:
                        dropped.add(col_i)
                        break

        self.dropped_columns = sorted(list(dropped))
        self.retained_columns = [c for c in cols if c not in dropped]
        self.correlated_pairs_ = pairs
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the selected subset to any dataset."""
        if self.retained_columns is None:
            raise RuntimeError("Filter must be fitted on training data before transform.")
        return X[self.retained_columns].copy()

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        """Fits and transforms training data."""
        return self.fit(X_train, y_train).transform(X_train)
