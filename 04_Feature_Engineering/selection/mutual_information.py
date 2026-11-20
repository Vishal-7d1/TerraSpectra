"""
04_Feature_Engineering/selection/mutual_information.py

Information-theoretic non-linear feature ranking:
Estimates Mutual Information (MI) between continuous features and discrete disease classes.
Strictly fit only on the training partition.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class LeakageFreeMutualInfoSelector:
    """Ranks features by Mutual Information on training split."""

    def __init__(self, top_k: int = 30, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.retained_columns: Optional[List[str]] = None
        self.ranking_df_: Optional[pd.DataFrame] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "LeakageFreeMutualInfoSelector":
        """Calculates MI scores on training observations."""
        mi_scores = mutual_info_classif(
            X_train,
            y_train,
            discrete_features=False,
            random_state=self.random_state,
        )

        ranking = pd.DataFrame({
            "feature": X_train.columns,
            "mutual_information": mi_scores,
        }).sort_values("mutual_information", ascending=False).reset_index(drop=True)

        self.ranking_df_ = ranking
        k = min(self.top_k, len(X_train.columns))
        self.retained_columns = list(ranking["feature"].iloc[:k].values)
        return self

    @property
    def selected_features_(self) -> Optional[List[str]]:
        return self.retained_columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.retained_columns is None:
            raise RuntimeError("Selector must be fitted on training data before transforming.")
        return X[self.retained_columns].copy()

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        return self.fit(X_train, y_train).transform(X_train)
