"""
04_Feature_Engineering/selection/statistical_selection.py

Univariate statistical hypothesis testing (ANOVA F-statistic):
Ranks and selects features based on between-class vs. within-class variance ratio.
Strictly fit only on the training partition.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


class LeakageFreeStatisticalSelector:
    """Ranks features by ANOVA F-statistic on training split."""

    def __init__(self, top_k: int = 30):
        self.top_k = top_k
        self.retained_columns: Optional[List[str]] = None
        self.ranking_df_: Optional[pd.DataFrame] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "LeakageFreeStatisticalSelector":
        """Fits ANOVA F-scores on training data."""
        f_scores, p_values = f_classif(X_train, y_train)

        # Replace NaNs if any
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        ranking = pd.DataFrame({
            "feature": X_train.columns,
            "f_statistic": f_scores,
            "p_value": p_values,
        }).sort_values("f_statistic", ascending=False).reset_index(drop=True)

        self.ranking_df_ = ranking
        k = min(self.top_k, len(X_train.columns))
        self.retained_columns = list(ranking["feature"].iloc[:k].values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.retained_columns is None:
            raise RuntimeError("Selector must be fitted on training data before transforming.")
        return X[self.retained_columns].copy()

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        return self.fit(X_train, y_train).transform(X_train)
