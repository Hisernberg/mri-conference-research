"""Population and completeness checks; synthetic metadata/metrics only."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_original_segmentation_exposure import derive_exposure, original_source_evidence, ROOT
from analyze_historical_sensitivity import select_frames, compare_saved_rows, check_core_completion
from combine_segmentation_studies import repeated_seed_summary


def metadata():
    development = ["train", "inner_first", "inner_late", "outer"]
    locked = ["locked_pure", "locked_alias", "locked_pair1", "locked_pair2"]
    old = {"development": development, "locked_test": locked, "folds": {
        "0": {"train": ["train"], "inner_val": ["inner_first", "inner_late"], "outer_val": ["outer"]},
        "1": {"train": ["outer", "inner_late"], "inner_val": ["train"], "outer_val": ["inner_first"]},
    }}
    mapping = {c: "g_" + c for c in development + locked}
    mapping["locked_alias"] = mapping["outer"]
    mapping["locked_pair1"] = mapping["locked_pair2"] = "g_pair"
    new = {"case_to_group": mapping, "development": ["train"],
           "locked_test": [c for c in development + locked if c != "train"]}
    return old, new


def test_historical_selection_excludes_entire_mixed_group_and_uses_recorded_order():
    old, new = metadata()
    result = derive_exposure(old, new, 1)
    assert result["sensitivity_group_ids"] == ["g_locked_pure", "g_pair"]
    assert result["sensitivity_full_case_ids"] == ["locked_pair1", "locked_pair2", "locked_pure"]
    rows = {r["case_id"]: r for r in result["rows"]}
    assert rows["locked_alias"]["original_fold0_role"] == "locked_test"
    assert not rows["locked_alias"]["old_locked_only_sensitivity"]
    assert rows["inner_first"]["original_validation_subset_direct"]
    assert not rows["inner_late"]["original_validation_subset_direct"]
    assert result["overlap"]["original_active_fold_training"]["reserved_group_overlap"] == 0


def test_current_partition_cannot_split_a_group():
    old, new = metadata()
    new["case_to_group"]["locked_pure"] = new["case_to_group"]["train"]
    with pytest.raises(ValueError, match="group overlap"):
        derive_exposure(old, new, 1)


def test_real_preserved_source_reconstructs_the_recorded_validation_scope():
    evidence = original_source_evidence(ROOT)
    assert evidence["logged_split_hash"] == "a392d0e39514f976"
    assert evidence["validation_case_limit"] == 16
    assert evidence["active_training_folds"] == [0]


def frames():
    return {seed: pd.DataFrame({"qmmf": [0.9, 0.2, 0.6], "control": [0.1, 0.3, 0.4]},
                              index=["excluded", "a", "b"]) for seed in [42, 43, 44]}


def test_sensitivity_retains_all_seeds_and_uses_group_means():
    selected = select_frames(frames(), ["a", "b", "excluded"], ["b", "a"],
                             ["qmmf", "control"], [42, 43, 44])
    assert set(selected) == {42, 43, 44}
    summary, differences, _ = repeated_seed_summary(selected, "fixture", draws=100)
    assert next(r["mean"] for r in summary if r["model"] == "qmmf") == pytest.approx(0.4)
    assert differences[0]["difference"] == pytest.approx(0.05)
    assert all(r["n_groups"] == 2 and r["n_training_seeds"] == 3 for r in summary)


def test_invalid_excluded_group_cannot_be_hidden_by_filtering():
    values = frames()
    values[42].loc["excluded", "qmmf"] = np.nan
    with pytest.raises(ValueError, match="Invalid Dice"):
        select_frames(values, ["a", "b", "excluded"], ["a", "b"], ["qmmf", "control"], [42, 43, 44])


@pytest.mark.parametrize("fault", ["missing_seed", "missing_model", "missing_group", "duplicate_group"])
def test_sensitivity_rejects_incomplete_or_ambiguous_source(fault):
    values = frames()
    if fault == "missing_seed":
        del values[44]
    elif fault == "missing_model":
        values[42] = values[42].drop(columns="control")
    elif fault == "missing_group":
        values[43] = values[43].drop(index="excluded")
    else:
        values[44] = pd.concat([values[44], values[44].loc[["a"]]])
    with pytest.raises(ValueError):
        select_frames(values, ["a", "b", "excluded"], ["a", "b"], ["qmmf", "control"], [42, 43, 44])


def test_core_summary_tampering_is_detected():
    rows = [{"model": "qmmf", "metric": "fixture", "mean": 0.4}]
    saved = pd.DataFrame(rows)
    compare_saved_rows(saved, rows, ["model", "metric"], ["mean"])
    saved.loc[0, "mean"] = 0.8
    with pytest.raises(ValueError, match="does not reproduce"):
        compare_saved_rows(saved, rows, ["model", "metric"], ["mean"])


def test_core_synthetic_or_incomplete_receipt_is_rejected():
    protocol = {"models": ["qmmf", "hemis", "no_quality", "unet25d"], "seeds": [42, 43, 44],
                "current_split_hash": "fixture", "main_source_sha256": "fixture", "bootstrap_draws": 10000}
    receipt = {"all_planned_completed": True, "completed_checkpoint_evaluations": 12,
               "models": protocol["models"], "seeds": protocol["seeds"], "split_hash": "fixture",
               "main_source_sha256": "fixture", "full_cases": 66, "full_groups": 51,
               "robustness_groups": 51, "subsets_per_group": 15, "bootstrap_draws": 10000,
               "locked_test_opened": True, "training_performed": False,
               "qualitative": {"verified": True, "synthetic_fixture": True}}
    with pytest.raises(ValueError, match="nonsynthetic"):
        check_core_completion(receipt, protocol)
    receipt["qualitative"]["synthetic_fixture"] = False
    check_core_completion(receipt, protocol)
    incomplete = deepcopy(receipt)
    incomplete["completed_checkpoint_evaluations"] = 11
    with pytest.raises(ValueError, match="Complete core"):
        check_core_completion(incomplete, protocol)
