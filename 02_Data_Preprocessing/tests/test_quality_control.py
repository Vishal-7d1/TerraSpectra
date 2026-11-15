"""
Tests for PreprocessingQualityControl.
"""

import numpy as np
from processors.quality_control import PreprocessingQualityControl


def test_quality_control_audit_pass():
    """Verifies that identical or faithfully smoothed data passes quality control."""
    raw = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    ])
    smoothed = raw + 0.001  # very low distortion
    labels = np.array(["Control", "Infected"])

    qc = PreprocessingQualityControl(max_allowed_rmse=0.05, min_correlation=0.98)
    report = qc.audit(raw, smoothed, labels, labels)

    assert report["overall_status"] == "PASSED"
    assert report["sample_counts"]["samples_preserved"] is True
    assert report["label_integrity"]["labels_aligned"] is True
    assert report["numerical_stability"]["is_stable"] is True
    assert report["spectral_fidelity"]["fidelity_passed"] is True


def test_quality_control_detects_label_mismatch():
    """Verifies that mismatched label arrays fail QC."""
    raw = np.ones((2, 4))
    proc = np.ones((2, 4))
    raw_labels = np.array(["Control", "Infected"])
    proc_labels = np.array(["Infected", "Control"])  # misaligned

    qc = PreprocessingQualityControl()
    report = qc.audit(raw, proc, raw_labels, proc_labels)
    assert report["label_integrity"]["labels_aligned"] is False
    assert report["overall_status"] == "FAILED"
