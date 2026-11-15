"""
Tests for MissingValueHandler and data cleaning operations.
"""

import numpy as np
from processors.missing_value_handler import MissingValueHandler


def test_missing_value_handler_audit():
    """Verifies that audit correctly detects NaNs, Infs, and out-of-range values."""
    mat = np.array([
        [0.1, 0.2, np.nan],
        [np.inf, -0.05, 0.4],
        [0.3, 1.25, 0.5],
    ])
    handler = MissingValueHandler(min_allowed=0.0, max_allowed=1.0)
    audit = handler.audit(mat)

    assert audit["nan_count"] == 1
    assert audit["inf_count"] == 1
    assert audit["negative_count"] == 1
    assert audit["over_max_count"] == 1
    assert audit["is_clean"] is False


def test_missing_value_handler_interpolation():
    """Verifies linear spectral interpolation removes NaNs and preserves bounds."""
    wls = np.array([500.0, 550.0, 600.0, 650.0])
    mat = np.array([
        [0.10, np.nan, 0.30, 0.40],
        [-0.05, 0.20, 0.35, 1.10],
    ])
    handler = MissingValueHandler(strategy="linear_spectral_interpolation", min_allowed=0.0, max_allowed=1.0)
    cleaned, report = handler.clean(mat, wls)

    assert np.isnan(cleaned).sum() == 0
    assert np.isinf(cleaned).sum() == 0
    # Linearly interpolated value between 0.10 and 0.30 should be 0.20
    assert cleaned[0, 1] == pytest.approx(0.20, 0.01)
    # Clipped values
    assert cleaned[1, 0] >= 0.0
    assert cleaned[1, 3] <= 1.0


import pytest
