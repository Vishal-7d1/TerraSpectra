"""
04_Feature_Engineering/tests/test_pipeline.py

Integration tests for the complete feature engineering pipeline on synthetic fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from extraction.vegetation_indices import VegetationIndexExtractor
from extraction.derivative_features import SpectralDerivativeExtractor
from extraction.spectral_features import SpectralShapeExtractor
from extraction.statistical_features import StatisticalFeatureExtractor
from validation.split_generator import StratifiedGroupSplitter
from validation.leakage_check import DataLeakageAuditor
from validation.feature_validation import FeatureQualityAuditor
from selection.variance_filter import LeakageFreeVarianceFilter
from selection.correlation_filter import LeakageFreeCorrelationFilter
from selection.model_importance import LeakageFreeModelImportanceSelector
from dimensionality_reduction.pca import LeakageFreePCA


def test_full_pipeline_synthetic_execution(mock_spectral_matrix, mock_wavelengths, mock_metadata_df):
    """Executes the full pipeline stages on synthetic data to ensure total structural coherence."""
    # 1. Feature Extraction
    vi_df, _ = VegetationIndexExtractor(mock_spectral_matrix, mock_wavelengths).extract_indices()
    deriv_df, _ = SpectralDerivativeExtractor(mock_spectral_matrix, mock_wavelengths).extract_derivative_features()
    shape_df, _ = SpectralShapeExtractor(mock_spectral_matrix, mock_wavelengths).extract_shape_features()
    stat_df, _ = StatisticalFeatureExtractor(mock_spectral_matrix, mock_wavelengths).extract_statistical_features()

    engineered_df = pd.concat([vi_df, deriv_df, shape_df, stat_df], axis=1)
    assert engineered_df.shape[0] == 24
    assert engineered_df.shape[1] == 10 + 8 + 11 + 22  # exactly 51 features

    # 2. Quality Audit
    qa = FeatureQualityAuditor(engineered_df).audit_quality()
    assert qa["is_valid"] is True

    # 3. Stratified Splitting
    splitter = StratifiedGroupSplitter(group_col="barcode", target_col="disease", random_state=42)
    split_df = splitter.generate_splits(mock_metadata_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    leakage = DataLeakageAuditor.audit_group_isolation(split_df, group_col="barcode")
    assert leakage["is_completely_isolated"] is True

    # 4. Fit Selection on Train
    train_mask = (split_df["split_partition"] == "train").values
    X_train = engineered_df.loc[train_mask]
    y_train = (mock_metadata_df.loc[train_mask, "disease"] == "Infected").astype(int).values

    var_f = LeakageFreeVarianceFilter(threshold=1e-5)
    X_train_v = var_f.fit_transform(X_train)

    corr_f = LeakageFreeCorrelationFilter(threshold=0.95)
    X_train_c = corr_f.fit_transform(X_train_v, y_train)

    top_k = min(5, X_train_c.shape[1])
    rf_sel = LeakageFreeModelImportanceSelector(n_estimators=10, top_k=top_k, random_state=42)
    X_train_final = rf_sel.fit_transform(X_train_c, y_train)

    assert X_train_final.shape[1] == top_k

    # 5. Transform Test without Leakage
    test_mask = (split_df["split_partition"] == "test").values
    X_test = engineered_df.loc[test_mask]
    X_test_final = rf_sel.transform(corr_f.transform(var_f.transform(X_test)))

    assert X_test_final.shape[1] == top_k
    assert list(X_test_final.columns) == list(X_train_final.columns)

    # 6. Fit PCA on Train
    pca = LeakageFreePCA(n_components=3, random_state=42)
    pca.fit(X_train)
    X_test_pca = pca.transform(X_test)
    assert X_test_pca.shape == (sum(test_mask), 3)
