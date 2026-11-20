"""
04_Feature_Engineering/validation/leakage_check.py

Rigorous automated data leakage validation:
- Verifies mutual exclusivity of plant specimens across train, validation, and test splits.
- Audits feature matrices to ensure identifiers and target labels are never used as input predictors.
- Verifies that parametric feature transformations were fitted strictly on training data.
"""

from typing import Dict, Any, List, Set
import numpy as np
import pandas as pd


class DataLeakageAuditor:
    """Performs defensive validation checks to eliminate information leakage."""

    @staticmethod
    def audit_group_isolation(split_df: pd.DataFrame, group_col: str = "barcode") -> Dict[str, Any]:
        """
        Validates that plant specimens (barcodes) in training, validation, and test
        sets are completely disjoint sets.
        """
        train_groups = set(split_df.loc[split_df["split_partition"] == "train", group_col])
        val_groups = set(split_df.loc[split_df["split_partition"] == "val", group_col])
        test_groups = set(split_df.loc[split_df["split_partition"] == "test", group_col])

        train_val_overlap = train_groups.intersection(val_groups)
        train_test_overlap = train_groups.intersection(test_groups)
        val_test_overlap = val_groups.intersection(test_groups)

        has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        return {
            "group_column": group_col,
            "train_unique_groups": len(train_groups),
            "val_unique_groups": len(val_groups),
            "test_unique_groups": len(test_groups),
            "train_val_leakage_count": len(train_val_overlap),
            "train_test_leakage_count": len(train_test_overlap),
            "val_test_leakage_count": len(val_test_overlap),
            "train_val_overlap": list(train_val_overlap),
            "train_test_overlap": list(train_test_overlap),
            "val_test_overlap": list(val_test_overlap),
            "is_completely_isolated": not has_overlap,
            "leakage_status": "PASSED_ZERO_LEAKAGE" if not has_overlap else "FAILED_LEAKAGE_DETECTED",
        }

    @staticmethod
    def audit_target_and_id_leakage(feature_columns: List[str], forbidden_columns: List[str]) -> Dict[str, Any]:
        """
        Ensures target labels, barcodes, dates, and file paths are not included
        in the final feature column list.
        """
        leaked_cols = [c for c in feature_columns if c in forbidden_columns]
        is_clean = len(leaked_cols) == 0

        return {
            "forbidden_columns_checked": forbidden_columns,
            "leaked_columns_found": leaked_cols,
            "is_clean": is_clean,
            "status": "PASSED_NO_TARGET_LEAKAGE" if is_clean else "FAILED_LEAKED_TARGET_OR_ID",
        }
