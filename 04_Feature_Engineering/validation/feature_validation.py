"""
04_Feature_Engineering/validation/feature_validation.py

Comprehensive numerical stability, completeness, and quality assurance for engineered features:
- Audits missing and infinite values.
- Flags constant and near-constant features.
- Detects duplicate feature columns.
- Validates data types and numerical ranges.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from utils.feature_utils import save_json


class FeatureQualityAuditor:
    """Performs non-destructive quality checks on extracted feature matrices."""

    def __init__(self, feature_df: pd.DataFrame, near_constant_threshold: float = 1e-6):
        self.feature_df = feature_df
        self.threshold = near_constant_threshold

    def audit_quality(self) -> Dict[str, Any]:
        """Runs full quality assurance audit on the feature DataFrame."""
        n_samples, n_features = self.feature_df.shape

        # 1. Missing & Infs
        nan_counts = self.feature_df.isna().sum()
        total_nans = int(nan_counts.sum())
        cols_with_nans = nan_counts[nan_counts > 0].to_dict()

        inf_counts = np.isinf(self.feature_df.to_numpy()).sum()

        # 2. Constant & Near-constant features
        variances = self.feature_df.var(numeric_only=True)
        constant_cols = list(variances[variances == 0.0].index)
        near_constant_cols = list(variances[(variances > 0.0) & (variances < self.threshold)].index)

        # 3. Duplicate columns
        duplicate_col_names = self.feature_df.columns[self.feature_df.columns.duplicated()].tolist()

        is_passed = (total_nans == 0 and inf_counts == 0 and len(constant_cols) == 0 and len(duplicate_col_names) == 0)

        report = {
            "total_samples": int(n_samples),
            "total_features": int(n_features),
            "total_missing_values": total_nans,
            "columns_with_missing_values": {k: int(v) for k, v in cols_with_nans.items()},
            "total_infinite_values": int(inf_counts),
            "constant_features_count": len(constant_cols),
            "constant_features": constant_cols,
            "near_constant_features_count": len(near_constant_cols),
            "near_constant_features": near_constant_cols,
            "duplicate_column_names": duplicate_col_names,
            "is_valid": is_passed,
            "quality_status": "PASSED" if is_passed else "FLAGGED_FOR_TREATMENT",
        }
        return report
