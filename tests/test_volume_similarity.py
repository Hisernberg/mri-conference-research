"""Validate volume-signature Pearson values against an independent reference."""
import importlib.util
from pathlib import Path
import numpy as np

SPEC = importlib.util.spec_from_file_location("volume_similarity",
    Path(__file__).resolve().parents[1] / "scripts/audit_segmentation_similarity.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_long_sparse_signature_correlations_are_bounded_and_match_reference():
    rng = np.random.default_rng(7)
    signatures = rng.normal(size=(4, 140400)).astype(np.float32)
    signatures[:, :100000] = 0
    signatures[1] = signatures[0] * np.float32(3.7)
    signatures[2] = -signatures[0]
    result = audit.pearson_matrix(signatures)
    np.testing.assert_allclose(result, np.corrcoef(signatures), rtol=0, atol=1e-12)
    assert result.min() >= -1 and result.max() <= 1
    assert abs(result[0, 1] - 1) < 1e-12
    assert abs(result[0, 2] + 1) < 1e-12
