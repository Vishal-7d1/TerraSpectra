"""
04_Feature_Engineering/tests/test_leakage.py

Unit tests for biological specimen grouping, stratified splitting, and defensive data leakage auditing.
"""

import pytest
import pandas as pd
from validation.split_generator import StratifiedGroupSplitter
from validation.leakage_check import DataLeakageAuditor


def test_stratified_group_splitter_zero_overlap(mock_metadata_df):
    """Verifies that plant barcodes never cross partition boundaries."""
    splitter = StratifiedGroupSplitter(group_col="barcode", target_col="disease", random_state=42)
    split_df = splitter.generate_splits(mock_metadata_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert "split_partition" in split_df.columns
    audit = DataLeakageAuditor.audit_group_isolation(split_df, group_col="barcode")

    assert audit["is_completely_isolated"] is True
    assert len(audit["train_val_overlap"]) == 0
    assert len(audit["train_test_overlap"]) == 0
    assert len(audit["val_test_overlap"]) == 0


def test_data_leakage_auditor_catches_intentional_leakage(mock_metadata_df):
    """Verifies that DataLeakageAuditor actively detects and flags intentional barcode overlap."""
    leaked_df = mock_metadata_df.copy()
    # Intentionally contaminate partitions: put PLANT_001 in both train and test
    leaked_df["split_partition"] = "train"
    leaked_df.loc[leaked_df["barcode"] == "PLANT_001", "split_partition"] = "test"
    # Overwrite one scan of PLANT_001 to train
    leaked_df.loc[leaked_df["scan_id"] == "PLANT_001_scan_1", "split_partition"] = "train"

    audit = DataLeakageAuditor.audit_group_isolation(leaked_df, group_col="barcode")
    assert audit["is_completely_isolated"] is False
    assert "PLANT_001" in audit["train_test_overlap"]
