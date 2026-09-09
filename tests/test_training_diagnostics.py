"""Training completion and convergence are different claims."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_segmentation import validate_training_history
from qmmf.config import ExperimentConfig


def fixture(scores, max_epochs=4, val_every=1, patience=2):
    cfg = ExperimentConfig().merged({"train.max_epochs": max_epochs, "train.val_every": val_every,
        "train.early_stopping_patience": patience, "train.steps_per_epoch": 5, "train.accumulation_steps": 2})
    frame = pd.DataFrame({"epoch": range(len(scores)), "total": .7, "epoch_seconds": 2.,
        "lr": cfg.train.lr, "selection_score": scores,
        "val_full_macro_dice": scores, "val_mean_subset_macro_dice": scores,
        "val_worst_subset_macro_dice": scores})
    selected = frame.dropna(subset=["selection_score"])
    index = selected.selection_score.idxmax()
    completion = {"best_epoch": int(frame.epoch[index]), "inner_selection_score": float(frame.selection_score[index])}
    return frame, cfg, completion


def test_last_epoch_best_is_reported_without_convergence_claim():
    frame,cfg,done=fixture([.2,.3,.4,.5])
    result=validate_training_history(frame,cfg,done)
    assert result["termination_reason"] == "epoch_budget_reached"
    assert result["best_at_last_validation"] is True
    assert result["last_three_validation_gain"] == pytest.approx(.2)
    assert result["scheduled_optimizer_step_opportunities"] == 12  # ceil accumulation, 3 per epoch
    assert "converged" not in result


def test_valid_patience_stop_and_first_tied_best():
    frame,cfg,done=fixture([.5,.5,.4],max_epochs=6)
    result=validate_training_history(frame,cfg,done)
    assert result["termination_reason"] == "validation_patience_reached"
    assert result["selected_epoch_one_based"] == 1
    assert result["trailing_nonimproving_validation_checks"] == 2


@pytest.mark.parametrize("damage,message", [
    ("composite", "Composite"), ("schedule", "Validation history"),
    ("epochs", "Training epochs"), ("premature", "stopped before"), ("ignored_patience", "continued after")])
def test_training_receipt_rejects_inconsistent_history(damage,message):
    frame,cfg,done=fixture([.2,.3,.4,.5])
    if damage == "composite":
        frame.loc[0,"val_full_macro_dice"] += .1
    elif damage == "schedule":
        frame.loc[0,"selection_score"] = np.nan
    elif damage == "epochs":
        frame.loc[0,"epoch"] = 1
    elif damage == "premature":
        frame,cfg,done=fixture([.2,.3],max_epochs=4)
    else:
        frame,cfg,done=fixture([.5,.4,.3,.6],max_epochs=4)
    with pytest.raises(ValueError,match=message):
        validate_training_history(frame,cfg,done)
