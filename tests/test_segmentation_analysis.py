"""Guard the independent result verifier's statistical unit and failure checks."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("segmentation_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_segmentation.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_case_mean_gives_equal_weight_with_different_empty_regions():
    # Case A's two evaluable regions score zero; B's three score one.
    # Equal case weighting must give .5, not the pooled region mean .6.
    frame = pd.DataFrame([
        {"case_id": cid, "region": region, "dice": float(cid == "b" or empty),
         "reference_empty": str(empty), "prediction_empty": str(cid == "a"),
         "volume_ref_ml": 0.0 if empty else 1.0,
         "volume_pred_ml": 0.0 if cid == "a" else 1.0}
        for cid in ["a", "b"] for region in ["wt", "tc", "et"]
        for empty in [cid == "a" and region == "et"]])
    _, macro = analysis.case_macros(frame, ["a", "b"])
    assert macro.mean() == .5
    with pytest.raises(ValueError, match="Evaluation cases"):
        analysis.case_macros(frame, ["a", "c"])
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.case_macros(pd.concat([frame, frame.iloc[:1]]), ["a", "b"])


def test_paired_bootstrap_preserves_constant_case_difference():
    control = np.array([.1, .3, .6, .7])
    matrix = pd.DataFrame({"qmmf": control + .15, "hemis": control})
    summaries, contrasts = analysis.paired_intervals(matrix, 500, 123, "macro_dice")
    assert len(summaries) == 2 and len(contrasts) == 1
    result = contrasts[0]
    np.testing.assert_allclose([result["difference"], result["ci_low"], result["ci_high"]], .15)
    assert result["n_cases"] == 4
    with pytest.raises(ValueError, match="missing or unaligned"):
        analysis.paired_intervals(matrix.assign(hemis=[.1, np.nan, .6, .7]), 50, 123, "macro_dice")


def test_duplicate_group_weighting_and_paired_intervals():
    mapping = {"a": "g0", "b": "g1", "c": "g1", "d": "g1"}
    full = analysis.aggregate_groups(pd.Series({"a": .2, "b": .8, "c": .8, "d": .8}), mapping)
    assert full.mean() == pytest.approx(.5)  # A case-level mean would be .65.
    table = pd.DataFrame({"qmmf": full, "control": full - .1})
    intervals, contrasts = analysis.paired_intervals(table, 100, 42, "dice", unit="group")
    assert intervals[0]["n_groups"] == 2 and "n_cases" not in intervals[0]
    np.testing.assert_allclose([contrasts[0][k] for k in ["difference", "ci_low", "ci_high"]], .1)
