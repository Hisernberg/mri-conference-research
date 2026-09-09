"""Metrics against hand-checkable toy cases, and loss-function behaviour.

The empty-reference tests matter most: an ET region with no reference voxels is
the single most common way a brain-tumour paper accidentally inflates its
headline Dice.
"""

import numpy as np
import pytest
import torch

from qmmf.calibration import (
    average_calibration_error, brier_score, expected_calibration_error,
    failure_detection_scores, risk_coverage_curve, selective_performance,
    TemperatureScaler,
)
from qmmf.losses import (
    QMMFLoss, boundary_loss, consistency_loss, ml1_ace_loss, nested_penalty,
    soft_dice_loss,
)
from qmmf.metrics import (
    dice_score, evaluate_case, hausdorff95, lesion_wise_scores, macro_dice,
    normalized_surface_dice, summarize,
)


# --------------------------------------------------------------------------- #
def test_dice_perfect_and_disjoint():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True
    assert dice_score(a, a)[0] == pytest.approx(1.0)
    b = np.zeros_like(a); b[7:9, 7:9, 7:9] = True
    assert dice_score(a, b)[0] == pytest.approx(0.0)


def test_dice_known_half_overlap():
    a = np.zeros(100, bool); a[:50] = True
    b = np.zeros(100, bool); b[25:75] = True
    # |A ∩ B| = 25, |A| + |B| = 100 -> Dice = 0.5
    assert dice_score(a, b)[0] == pytest.approx(0.5)


def test_empty_reference_and_empty_prediction_is_flagged():
    empty = np.zeros((4, 4, 4), bool)
    d, ref_empty, pred_empty = dice_score(empty, empty)
    assert d == 1.0 and ref_empty and pred_empty


def test_empty_reference_with_false_positive_scores_zero():
    empty = np.zeros((4, 4, 4), bool)
    pred = empty.copy(); pred[0, 0, 0] = True
    d, ref_empty, pred_empty = dice_score(pred, empty)
    assert d == 0.0 and ref_empty and not pred_empty


def test_macro_dice_excludes_empty_reference_regions():
    """An ET region with no reference must not contribute a free 1.0."""
    pred = np.zeros((3, 8, 8, 8), np.uint8)
    ref = np.zeros((3, 8, 8, 8), np.uint8)
    ref[0, 1:5, 1:5, 1:5] = 1; pred[0, 1:5, 1:5, 1:5] = 1     # perfect WT
    ref[1, 2:4, 2:4, 2:4] = 1; pred[1, 2:4, 2:4, 2:4] = 1     # perfect TC
    # ET is empty in both.
    rows = evaluate_case(pred, ref, (1, 1, 1), "case", lesion_metrics=False)
    assert macro_dice(rows, exclude_empty_reference=True) == pytest.approx(1.0)
    assert [r.reference_empty for r in rows] == [False, False, True]

    # And a half-wrong WT must pull the macro down, not be masked by the free ET.
    pred[0, 1:3] = 0
    rows2 = evaluate_case(pred, ref, (1, 1, 1), "case", lesion_metrics=False)
    assert macro_dice(rows2, exclude_empty_reference=True) < 1.0


def test_summary_counts_empty_reference_cases():
    pred = np.zeros((3, 6, 6, 6), np.uint8)
    ref = np.zeros((3, 6, 6, 6), np.uint8)
    ref[0, 1:4, 1:4, 1:4] = 1; pred[0, 1:4, 1:4, 1:4] = 1
    pred[2, 0, 0, 0] = 1                      # ET false positive on empty ref
    rows = evaluate_case(pred, ref, (1, 1, 1), "c", lesion_metrics=False)
    s = summarize(rows)
    assert s["n_empty_reference_et"] == 1
    assert s["n_empty_ref_false_positive_et"] == 1


def test_hd95_and_nsd_use_spacing():
    a = np.zeros((20, 20, 20), bool); a[5:15, 5:15, 5:15] = True
    b = np.zeros_like(a); b[5:15, 5:15, 5:16] = True     # one voxel further
    iso = hausdorff95(b, a, (1.0, 1.0, 1.0))
    anis = hausdorff95(b, a, (1.0, 1.0, 3.0))
    assert anis > iso, "HD95 must scale with physical spacing"


def test_nsd_is_one_for_identical_masks():
    a = np.zeros((12, 12, 12), bool); a[3:9, 3:9, 3:9] = True
    assert normalized_surface_dice(a, a, (1, 1, 1), 1.0) == pytest.approx(1.0)


def test_lesion_scores_on_two_lesions_one_detected():
    ref = np.zeros((20, 20, 20), bool)
    ref[2:8, 2:8, 2:8] = True          # lesion 1
    ref[12:18, 12:18, 12:18] = True    # lesion 2
    pred = np.zeros_like(ref)
    pred[2:8, 2:8, 2:8] = True         # only lesion 1 found
    out = lesion_wise_scores(pred, ref)
    assert out["recall"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
def test_soft_dice_is_low_for_perfect_prediction():
    target = torch.zeros(2, 3, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    logits = (target * 20.0) - 10.0
    assert soft_dice_loss(logits, target).item() < 0.02


def test_nested_penalty_is_zero_for_valid_hierarchy_and_positive_otherwise():
    valid = torch.stack([
        torch.full((1, 4, 4), 5.0), torch.full((1, 4, 4), 0.0),
        torch.full((1, 4, 4), -5.0),
    ], dim=1)
    assert nested_penalty(valid).item() == pytest.approx(0.0, abs=1e-3)
    invalid = torch.flip(valid, dims=[1])       # ET most confident: violates nesting
    assert nested_penalty(invalid).item() > 0.1


def test_boundary_loss_returns_zero_gradientless_value_on_empty_target():
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 3, 8, 8)
    value = boundary_loss(logits, target)
    assert value.item() == 0.0
    value.backward()          # must not raise or produce NaN
    assert torch.isfinite(logits.grad).all()


def test_consistency_only_applies_where_teacher_is_confident():
    student = torch.zeros(1, 3, 4, 4)
    unconfident_teacher = torch.zeros(1, 3, 4, 4)     # sigmoid = 0.5 everywhere
    assert consistency_loss(student, unconfident_teacher, 0.9).item() == 0.0
    confident_teacher = torch.full((1, 3, 4, 4), 5.0)
    assert consistency_loss(student, confident_teacher, 0.9).item() > 0.0


def test_total_loss_includes_declared_components_only():
    from qmmf.config import LossConfig
    cfg = LossConfig(consistency=0.0, calibration=0.0)
    loss = QMMFLoss(cfg)
    logits = torch.randn(2, 3, 16, 16)
    target = (torch.rand(2, 3, 16, 16) > 0.6).float()
    parts = loss(logits, target)
    assert "consistency" not in parts and "calibration" not in parts
    assert {"dice", "bce", "seg", "boundary", "nested", "total"} <= set(parts)
    assert torch.isfinite(parts["total"])


def test_ml1_ace_is_lower_for_a_calibrated_predictor():
    torch.manual_seed(0)
    n = 20000
    p = torch.rand(n)
    y = (torch.rand(n) < p).float()             # perfectly calibrated by construction
    calibrated = torch.log(p / (1 - p))
    overconfident = calibrated * 3.0
    assert ml1_ace_loss(calibrated, y).item() < ml1_ace_loss(overconfident, y).item()


# --------------------------------------------------------------------------- #
def test_calibration_metrics_prefer_the_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    over = np.clip(p * 1.6, 0, 1)
    assert expected_calibration_error(p, y) < expected_calibration_error(over, y)
    assert average_calibration_error(p, y) < average_calibration_error(over, y)
    assert brier_score(p, y) < brier_score(over, y)


def test_temperature_scaling_moves_towards_calibration():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    logits = np.log(p / (1 - p))[None, :] * 2.5        # deliberately overconfident
    scaler = TemperatureScaler.fit(logits, y[None, :], regions=("wt",))
    before = expected_calibration_error(1 / (1 + np.exp(-logits[0])), y)
    after = expected_calibration_error(scaler.transform_probs(logits)[0], y)
    assert after < before
    assert scaler.temperatures[0] > 1.0       # it should cool the predictions


def test_risk_coverage_decreases_with_useful_uncertainty():
    rng = np.random.default_rng(2)
    dice = rng.random(200)
    uncertainty = 1.0 - dice                  # perfectly informative
    curve = risk_coverage_curve(dice, uncertainty)
    assert curve["risk"][0] <= curve["risk"][-1]
    sel = selective_performance(dice, uncertainty, (1.0, 0.8))
    assert sel["score_at_coverage_80"] > sel["score_at_coverage_100"]


def test_failure_detection_scores_handle_degenerate_labels():
    out = failure_detection_scores([0.1, 0.2, 0.3], [0, 0, 0])
    assert np.isnan(out["auroc"])             # no failures: undefined, not 0.5
