"""
04_Feature_Engineering/tests/test_feature_selection.py

Unit tests for leakage-free feature selection algorithms (variance, correlation, RF importance, MI, ANOVA).
"""

import pytest
import numpy as np
import pandas as pd
from selection.variance_filter import LeakageFreeVarianceFilter
from selection.correlation_filter import LeakageFreeCorrelationFilter
from selection.model_importance import LeakageFreeModelImportanceSelector
from selection.mutual_information import LeakageFreeMutualInfoSelector
from selection.statistical_selection import LeakageFreeStatisticalSelector


def test_variance_filter_removes_constant(mock_feature_df):
    """Verifies that the variance filter removes zero or sub-threshold variance features."""
    v_filter = LeakageFreeVarianceFilter(threshold=1e-4)
    transformed_df = v_filter.fit_transform(mock_feature_df)

    assert "feat_constant" in v_filter.dropped_columns
    assert "feat_constant" not in transformed_df.columns
    assert transformed_df.shape[1] == mock_feature_df.shape[1] - 1


def test_correlation_filter_removes_collinear(mock_feature_df):
    """Verifies that the correlation filter removes features with |r| > threshold."""
    # First remove constant so correlation is well-defined
    df_clean = mock_feature_df.drop(columns=["feat_constant"])
    y = (df_clean["feat_norm_1"] > 10).astype(int).values

    c_filter = LeakageFreeCorrelationFilter(threshold=0.90)
    transformed_df = c_filter.fit_transform(df_clean, y)

    assert len(c_filter.dropped_columns) >= 1
    # Either feat_collinear_1 or feat_norm_1 was dropped
    assert ("feat_collinear_1" in c_filter.dropped_columns) or ("feat_norm_1" in c_filter.dropped_columns)
    assert transformed_df.shape[1] < df_clean.shape[1]


def test_model_importance_selector_top_k(mock_feature_df):
    """Verifies that Random Forest selector correctly selects specified top K features."""
    df_clean = mock_feature_df.drop(columns=["feat_constant"])
    y = (df_clean["feat_norm_1"] > 10).astype(int).values

    rf_sel = LeakageFreeModelImportanceSelector(n_estimators=20, top_k=3, random_state=42)
    transformed = rf_sel.fit_transform(df_clean, y)

    assert transformed.shape[1] == 3
    assert len(rf_sel.ranking_df_) == df_clean.shape[1]
    assert rf_sel.ranking_df_["importance"].is_monotonic_decreasing


def test_mutual_info_and_anova_selectors(mock_feature_df):
    """Verifies that MI and ANOVA F-statistic ranking components execute and produce valid rankings."""
    df_clean = mock_feature_df.drop(columns=["feat_constant"])
    y = (df_clean["feat_norm_1"] > 10).astype(int).values

    mi_sel = LeakageFreeMutualInfoSelector(top_k=2).fit(df_clean, y)
    assert len(mi_sel.selected_features_) == 2
    assert len(mi_sel.ranking_df_) == df_clean.shape[1]

    f_sel = LeakageFreeStatisticalSelector(top_k=2).fit(df_clean, y)
    assert len(f_sel.selected_features_) == 2
    assert len(f_sel.ranking_df_) == df_clean.shape[1]
