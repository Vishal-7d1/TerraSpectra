"""
04_Feature_Engineering/selection/model_importance.py

Tree-based ensemble feature importance ranking:
Fits a Random Forest Classifier strictly on the training partition
to rank features by mean decrease in impurity (Gini importance).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class LeakageFreeModelImportanceSelector:
    """Ranks features using ensemble tree importance on training split."""

    def __init__(self, n_estimators: int = 100, top_k: int = 30, random_state: int = 42):
        self.n_estimators = n_estimators
        self.top_k = top_k
        self.random_state = random_state
        self.retained_columns: Optional[List[str]] = None
        self.ranking_df_: Optional[pd.DataFrame] = None
        self.model_: Optional[RandomForestClassifier] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "LeakageFreeModelImportanceSelector":
        """Fits Random Forest strictly on training data."""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.model_ = rf

        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

        ranking = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importances,
            "importance_std": std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

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
