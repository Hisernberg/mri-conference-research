"""Protected evaluation must preserve cohorts, checkpoints, and every fixed seed.

All scores below are artificial test fixtures; no protected MRI is loaded.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
import locked_evaluate as locked
import analyze_locked_segmentation as analysis
from conference_run import VARIANTS
from qmmf.config import ExperimentConfig
from qmmf.quality import QualityNormalizer
from qmmf.shapley import shapley_from_subset_table
from qmmf.splits import Splits
from qmmf.subsets import ALL_SUBSET_KEYS, subset_key


@pytest.fixture
def protocol():
    return locked.read(locked.FROZEN / "locked_evaluation_protocol.json")


@pytest.fixture
def splits():
    return Splits.load(locked.FROZEN / "splits.json")


def test_protected_representatives_and_role_boundary(protocol, splits):
    locked.validate_protocol(protocol, splits)
    changed = copy.deepcopy(protocol)
    changed["full_modality_case_ids"][0] = splits.folds["0"]["train"][0]
    with pytest.raises(ValueError, match="full cohort"):
        locked.validate_protocol(changed, splits)
    changed = copy.deepcopy(protocol)
    original = set(changed["robustness_case_ids"])
    alternate = next(c for c in splits.locked_test if c not in original)
    group = splits.case_to_group[alternate]
    index = next(i for i,c in enumerate(changed["robustness_case_ids"]) if splits.case_to_group[c] == group)
    changed["robustness_case_ids"][index] = alternate
    with pytest.raises(ValueError, match="hash rule"):
        locked.validate_protocol(changed, splits)


def make_normalizer(splits):
    ids = splits.folds["0"]["train"]
    normalizer = QualityNormalizer.fit([np.ones((4,7), dtype=np.float32)])
    return {"training_case_ids": ids, "training_group_ids": sorted({splits.case_to_group[c] for c in ids}),
            "normalizer": normalizer.to_dict()}


@pytest.mark.parametrize("damage", ["epoch", "normalizer", "cohort", "weights", "config"])
def test_rejects_changed_selected_checkpoint(splits, damage):
    cfg = ExperimentConfig()
    norm = make_normalizer(splits)
    done = {"config_hash": cfg.config_hash(), "best_epoch": 19, "inner_selection_score": .5}
    payload = {"config_hash": cfg.config_hash(), "config": cfg.to_dict(),
               "state": {"best_epoch": 19, "epoch": 20, "best_score": .5},
               "quality_normalizer": copy.deepcopy(norm["normalizer"]), "model": {"w": torch.ones(2)}}
    locked.validate_checkpoint(payload, cfg, done, norm, splits)
    if damage == "epoch":
        payload["state"]["epoch"] = 30
    elif damage == "normalizer":
        payload["quality_normalizer"]["median"][0][0] = 999
    elif damage == "cohort":
        norm["training_case_ids"] = norm["training_case_ids"] + splits.locked_test[:1]
    elif damage == "weights":
        payload["model"]["w"][0] = float("nan")
    else:
        payload["config"]["seed"] = 99
    with pytest.raises(ValueError):
        locked.validate_checkpoint(payload, cfg, done, norm, splits)


def synthetic_run(folder, protocol, splits, name="qmmf", seed=42):
    folder.mkdir(parents=True)
    model, ablation = VARIANTS[name]
    cfg = ExperimentConfig().merged({"seed": seed, "train.seed": seed, "model": model, "ablation": ablation,
        "data.split_hash": splits.split_hash, "train.max_epochs": 120,
        "train.steps_per_epoch": 100, "train.val_every": 10})
    ids = protocol["full_modality_case_ids"]
    # Deliberately vary the number of cases per group and use an empty region.
    # Only test groups are needed, map those to [0.25, 0.45].
    group_scores = {g: .25+i*.004 for i,g in enumerate(sorted({splits.case_to_group[c] for c in ids}))}
    offset = {"qmmf": .1, "hemis": 0., "no_quality": .04, "unet25d": .02}[name]+(seed-42)*.005
    macros = {c: group_scores[splits.case_to_group[c]] + offset for c in ids}
    frame = pd.DataFrame([{"case_id": c, "region": region, "dice": (1. if empty else macros[c]),
        "reference_empty": empty, "prediction_empty": empty,
        "volume_ref_ml": float(not empty), "volume_pred_ml": float(not empty)}
        for c in ids for region in ["wt", "tc", "et"] for empty in [region == "et" and c == ids[0]]])
    frame.to_csv(folder / "full_case_metrics.csv", index=False)
    grouped = analysis.aggregate_groups(pd.Series(macros), splits.case_to_group)
    locked.save(folder / "full_summary.json", {"per_case_macro": macros, "per_group_macro": grouped.to_dict(),
        "primary_group_macro_dice": grouped.mean(), "summary": {"macro_dice": np.mean(list(macros.values()))}})
    table = {c: {k: macros[c] if k == subset_key(locked.MODALITIES) else macros[c]-.1 for k in ALL_SUBSET_KEYS}
             for c in protocol["robustness_case_ids"]}
    regions = {c: {r: {k: 1. if (r == "et" and c == ids[0]) else v for k,v in vals.items()}
                   for r in ["wt", "tc", "et"]} for c,vals in table.items()}
    subsets = {"per_case": table, "per_case_region": regions,
        "mean_over_subsets": {c: np.mean(list(v.values())) for c,v in table.items()},
        "worst_subset": {c: min(v.values()) for c,v in table.items()},
        "shapley": {c: shapley_from_subset_table(v) for c,v in table.items()}}
    locked.save(folder / "modality_subsets.json", subsets)
    parent = {"completed": True, "locked_test_opened": False, "name": name, "seed": seed,
        "best_epoch": 19, "inner_selection_score": .5, "config_hash": cfg.config_hash()}
    provenance = {"model": name, "seed": seed, "parent_source_sha256": locked.MAIN_SOURCE_SHA,
        "training_performed": False, "normalizer_fitted": False,
        "checkpoint_sha256": "a"*64, "parent_config_hash": cfg.config_hash(),
        "config": cfg.to_dict(), "normalizer": make_normalizer(splits), "parent_completion": parent,
        "selected_epoch": 19, "inner_selection_score": .5,
        "frozen_protocol_sha256": analysis.json_sha(protocol)}
    locked.save(folder / "checkpoint_provenance.json", provenance)
    locked.save(folder / "completed.json", {"completed": True, "locked_test_opened": True,
        "scope": "protected_image_groups", "name": name, "seed": seed,
        "parent_config_hash": cfg.config_hash(), "checkpoint_sha256": "a"*64,
        "full_group_macro_dice": grouped.mean(), "mean_subset_macro_dice": np.mean(list(subsets["mean_over_subsets"].values())),
        "worst_subset_macro_dice": np.mean(list(subsets["worst_subset"].values())),
        "full_cases": len(ids), "full_groups": 51, "robustness_groups": 51, "subsets_per_group": 15})
    stem = f"qmmf_runs/{cfg.run_id()}"
    return [{"path": path, "sha256": analysis.json_sha(data)} for path,data in [
        (f"configs/{name}.json", cfg.to_dict()), (f"{stem}/completed.json", parent),
        (f"{stem}/quality_normalizer.json", provenance["normalizer"])]]


def test_protected_verifier_recomputes_regions_and_group_weighting(tmp_path, protocol, splits):
    run = tmp_path / "run"; synthetic_run(run, protocol, splits)
    result = analysis.verify_one(run, protocol, splits)
    assert result["full_group_macro_dice"].mean() == pytest.approx(.45)
    assert len(result["subsets"]) == 51*15
    subset = locked.read(run / "modality_subsets.json")
    cid = protocol["robustness_case_ids"][0]
    subset["per_case"][cid][ALL_SUBSET_KEYS[0]] += .01
    locked.save(run / "modality_subsets.json", subset)
    with pytest.raises(ValueError, match="regional macro"):
        analysis.verify_one(run, protocol, splits)


def test_complete_protected_matrix_and_all_fixed_seeds(tmp_path, protocol, splits, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "analysis"
    source.mkdir()
    parents = {str(s): {"artifact_hashes": []} for s in locked.SEEDS}
    for seed in locked.SEEDS:
        for name in locked.MODELS:
            parents[str(seed)]["artifact_hashes"].extend(synthetic_run(source / "runs" / f"{name}_{seed}", protocol, splits, name, seed))
    locked.save(source / "frozen_protocol.json", protocol)
    splits.save(source / "splits.json")
    locked.save(source / "evaluation_protocol.json", {**protocol, "locked_test_opened": True,
        "main_source_sha256": locked.MAIN_SOURCE_SHA, "training_performed": False, "normalizer_fitted": False})
    locked.save(source / "development_gate.json", {"main_source_sha256": locked.MAIN_SOURCE_SHA,
        "frozen_protocol_sha256": analysis.json_sha(protocol), "parents": parents,
        "combination_verification": {"all_required_completed": True, "completed_model_fits": 27,
                                     "seeds": locked.SEEDS, "locked_test_opened": False}})
    locked.save(source / "access_record.json", {"locked_test_opened": True,
        "frozen_protocol_sha256": analysis.json_sha(protocol),
        "development_gate_sha256": analysis.json_sha(locked.read(source / "development_gate.json"))})
    matrix = [{"seed": s, "name": n} for s in locked.SEEDS for n in locked.MODELS]
    locked.save(source / "session_status.json", {"planned": matrix, "completed": matrix, "all_planned_completed": True})
    monkeypatch.setattr(analysis, "plot", lambda *args: None)
    result = analysis.analyze(source, dest, draws=300)
    assert result["completed_checkpoint_evaluations"] == 12
    deltas = pd.read_csv(dest / "paired_repeated_seed_deltas.csv")
    assert deltas.n_groups.eq(51).all() and deltas.n_training_seeds.eq(3).all()
    assert deltas.comparison_family_size.eq(3).all()
    np.testing.assert_allclose(deltas[deltas.control == "hemis"]["difference"], .1)
    (source / "runs/qmmf_44/completed.json").unlink()
    with pytest.raises(ValueError, match="matrix is incomplete"):
        analysis.analyze(source, tmp_path / "missing_seed", draws=20)
