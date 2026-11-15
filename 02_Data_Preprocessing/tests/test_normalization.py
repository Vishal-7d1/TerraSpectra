"""
Tests for SpectralNormalizer and data leakage prevention.
"""

import numpy as np
import pytest
from processors.spectral_correction import SpectralNormalizer


def test_snv_normalization_properties():
    """Verifies that SNV normalizes each sample across spectral bands to mean 0 and std 1."""
    np.random.seed(42)
    mat = np.random.uniform(0.1, 0.8, size=(8, 50))
    normalizer = SpectralNormalizer(method="snv")
    norm = normalizer.fit_transform(mat)

    assert norm.shape == mat.shape
    sample_means = np.mean(norm, axis=1)
    sample_stds = np.std(norm, axis=1)

    assert np.allclose(sample_means, 0.0, atol=1e-5)
    assert np.allclose(sample_stds, 1.0, atol=1e-5)


def test_fit_transform_prevents_data_leakage():
    """
    Critical Test: Verifies that StandardScaler parameters are learned ONLY from train data
    and applied to test data without re-fitting.
    """
    X_train = np.array([
        [10.0, 20.0],
        [20.0, 40.0],
    ])
    X_test = np.array([
        [50.0, 100.0]
    ])

    scaler = SpectralNormalizer(method="standard")
    scaler.fit(X_train)

    # Learned params: mean should be [15.0, 30.0], std [5.0, 10.0]
    assert np.allclose(scaler.params_["mean"], [15.0, 30.0])
    assert np.allclose(scaler.params_["std"], [5.0, 10.0])

    # Transform test set using TRAIN parameters
    X_test_trans = scaler.transform(X_test)
    expected_test_trans = np.array([[ (50 - 15) / 5, (100 - 30) / 10 ]])
    assert np.allclose(X_test_trans, expected_test_trans)


def test_unfitted_transform_raises_error():
    """Verifies that calling transform() without fit() raises RuntimeError."""
    scaler = SpectralNormalizer(method="standard")
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))
