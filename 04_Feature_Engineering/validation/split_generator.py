"""
04_Feature_Engineering/validation/split_generator.py

Leakage-safe stratified group dataset partitioning:
- Groups observations strictly by physical plant identifier (barcode).
- Stratifies plant groups by target disease status (Control vs. Infected).
- Partitions into Train (70%), Validation (15%), and Test (15%) splits.
- Guarantees 0% sample or specimen leakage across partitions.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from utils.feature_utils import save_json


class StratifiedGroupSplitter:
    """Generates biologically isolated train, validation, and test partitions."""

    def __init__(self, group_col: str = "barcode", target_col: str = "disease", random_state: int = 42):
        self.group_col = group_col
        self.target_col = target_col
        self.random_state = random_state

    def generate_splits(
        self,
        metadata_df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> pd.DataFrame:
        """
        Splits unique plant barcodes into train/val/test partitions while preserving target class proportions.
        Maps assignments back to individual observations.
        """
        if self.group_col not in metadata_df.columns:
            raise KeyError(f"Group column '{self.group_col}' missing from metadata.")
        if self.target_col not in metadata_df.columns:
            raise KeyError(f"Target column '{self.target_col}' missing from metadata.")

        # Aggregate unique plants and their disease label
        plants_df = metadata_df[[self.group_col, self.target_col]].drop_duplicates().reset_index(drop=True)
        
        # 1. First split: Train (70%) vs Temp (30%)
        temp_ratio = val_ratio + test_ratio
        train_plants, temp_plants = train_test_split(
            plants_df,
            test_size=temp_ratio,
            stratify=plants_df[self.target_col],
            random_state=self.random_state,
        )

        # 2. Second split: Temp into Validation (50% of temp = 15%) and Test (50% of temp = 15%)
        val_sub_ratio = val_ratio / temp_ratio
        val_plants, test_plants = train_test_split(
            temp_plants,
            test_size=(1.0 - val_sub_ratio),
            stratify=temp_plants[self.target_col],
            random_state=self.random_state,
        )

        train_barcodes = set(train_plants[self.group_col])
        val_barcodes = set(val_plants[self.group_col])
        test_barcodes = set(test_plants[self.group_col])

        # Map to each observation
        split_series = []
        for bc in metadata_df[self.group_col]:
            if bc in train_barcodes:
                split_series.append("train")
            elif bc in val_barcodes:
                split_series.append("val")
            elif bc in test_barcodes:
                split_series.append("test")
            else:
                split_series.append("unassigned")

        split_df = pd.DataFrame({
            "sample_index": np.arange(len(metadata_df)),
            self.group_col: metadata_df[self.group_col].values,
            self.target_col: metadata_df[self.target_col].values,
            "split_partition": split_series,
        })

        return split_df

    def audit_splits(self, split_df: pd.DataFrame) -> Dict[str, Any]:
        """Audits partition counts, unique plant barcodes, and class balance."""
        audit = {}
        for part in ["train", "val", "test"]:
            sub = split_df[split_df["split_partition"] == part]
            audit[part] = {
                "total_observations": int(len(sub)),
                "unique_plant_barcodes": int(sub[self.group_col].nunique()),
                "class_counts": sub[self.target_col].value_counts().to_dict(),
                "percentage_of_total": round((len(sub) / len(split_df)) * 100, 2),
            }
        return audit
