"""
04_Feature_Engineering/tests/test_pca.py

Unit tests for LeakageFreePCA dimensionality reduction and pipeline serialization.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from dimensionality_reduction.pca import LeakageFreePCA


def test_pca_fit_transform_shape(mock_feature_df):
    """Verifies that PCA transforms input to exact number of requested principal components."""
    clean_df = mock_feature_df.drop(columns=["feat_constant"])
    n_comp = 3

    pca_mod = LeakageFreePCA(n_components=n_comp, random_state=42)
    scores = pca_mod.fit_transform(clean_df)

    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (len(clean_df), n_comp)
    assert list(scores.columns) == [f"PCA_component_{i+1}" for i in range(n_comp)]
    assert np.isclose(np.sum(pca_mod.pca.explained_variance_ratio_), np.sum(pca_mod.pca.explained_variance_ratio_))


def test_pca_serialization_roundtrip(mock_feature_df):
    """Verifies that the fitted PCA model can be serialized and deserialized identically."""
    clean_df = mock_feature_df.drop(columns=["feat_constant"])
    pca_mod = LeakageFreePCA(n_components=2, random_state=42)
    scores_original = pca_mod.fit_transform(clean_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path = Path(tmpdir) / "fitted_pca.joblib"
        pca_mod.save_fitted_pipeline(joblib_path)
        assert joblib_path.exists()

        loaded_pca = LeakageFreePCA.load_fitted_pipeline(joblib_path)
        scores_loaded = loaded_pca.transform(clean_df)

        pd.testing.assert_frame_equal(scores_original, scores_loaded)
