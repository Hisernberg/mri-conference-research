#!/usr/bin/env python3
"""Describe verified reserved results without new inference or hypothesis tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_segmentation import read, require, close
from report_segmentation_study import summarize_regions
from qmmf.config import MODALITIES
from qmmf.subsets import ALL_SUBSET_KEYS, subset_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(source, dest):
    source, dest = Path(source), Path(dest)
    receipt = read(source / "verification.json")
    require(receipt.get("all_planned_completed") and receipt["completed_checkpoint_evaluations"] == 12,
            "Complete reserved verification is required")
    require(receipt["qualitative"].get("verified") and
            not receipt["qualitative"].get("synthetic_fixture", True), "Actual verified figures are required")
    require(receipt["full_cases"] == 66 and receipt["full_groups"] == 51 and
            receipt["robustness_groups"] == 51 and receipt["subsets_per_group"] == 15,
            "Reserved cohort differs")
    models, seeds = receipt["models"], receipt["seeds"]
    require(set(models) == {"qmmf", "hemis", "no_quality", "unet25d"} and seeds == [42, 43, 44],
            "Model or seed matrix differs")
    input_names = ["verification.json", "splits.json", "frozen_protocol.json", "session_status.json",
                   "all_full_case_metrics.csv", "region_and_empty_case_metrics.csv",
                   "all_subset_case_metrics.csv", "shapley_by_case_seed.csv", "all_evaluation_runs.csv"]
    inputs = {name: sha(source / name) for name in input_names}
    splits, protocol = read(source / "splits.json"), read(source / "frozen_protocol.json")
    group_map = splits["case_to_group"]
    full_ids, robust_ids = set(protocol["full_modality_case_ids"]), set(protocol["robustness_case_ids"])
    cases = pd.read_csv(source / "all_full_case_metrics.csv")
    require(len(cases) == 12 * 66 * 3 and
            not cases.duplicated(["model", "seed", "case_id", "region"]).any(),
            "Full regional rows are missing or duplicated")
    regional = pd.read_csv(source / "region_and_empty_case_metrics.csv")
    require(len(regional) == 36 and not regional.duplicated(["model", "seed", "region"]).any(),
            "Regional summary coverage differs")
    regional = regional.set_index(["model", "seed", "region"])
    rows = []
    for model in models:
        for seed in seeds:
            for region in ["wt", "tc", "et"]:
                frame = cases[(cases.model == model) & (cases.seed == seed) & (cases.region == region)].copy()
                require(set(frame.case_id) == full_ids, "Full-case cohort differs")
                frame["group_id"] = frame.case_id.map(group_map)
                nonempty = frame[~frame.reference_empty]
                value = nonempty.groupby("group_id").dice.mean().mean()
                saved = regional.loc[(model, seed, region)]
                close(value, saved.group_dice_nonempty_reference, "Regional group Dice differs")
                counts = {"n_nonempty_reference": len(nonempty),
                          "n_empty_reference": int(frame.reference_empty.sum()),
                          "n_empty_predictions": int(frame.prediction_empty.sum()),
                          "n_empty_reference_false_positives": int((frame.reference_empty & ~frame.prediction_empty).sum())}
                require(all(saved[k] == v for k, v in counts.items()), "Regional empty-case counts differ")
                rows.append({"model": model, "seed": seed, "region": region,
                             "group_dice_nonempty_reference": value,
                             "n_groups_with_nonempty_reference": nonempty.group_id.nunique(), **counts})
    regions = pd.DataFrame(rows)
    regional_summary = summarize_regions(regions, models, seeds)

    subsets = pd.read_csv(source / "all_subset_case_metrics.csv")
    require(len(subsets) == 12 * 51 * 15 and
            not subsets.duplicated(["model", "seed", "case_id", "subset"]).any(),
            "Subset rows are missing or duplicated")
    values = subsets.macro_dice.to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid subset Dice")
    require(subsets.group_id.equals(subsets.case_id.map(group_map)), "Subset group identity differs")
    subset_seed_rows, shapley_seed_rows, costs = [], [], []
    shapley = pd.read_csv(source / "shapley_by_case_seed.csv")
    require(len(shapley) == 12 * 51 and not shapley.duplicated(["model", "seed", "case_id"]).any(),
            "Shapley rows are missing or duplicated")
    ledger = pd.read_csv(source / "all_evaluation_runs.csv")
    require(len(ledger) == 12 and not ledger.duplicated(["name", "seed"]).any() and ledger.completed.all(),
            "Evaluation ledger differs")
    for model in models:
        for seed in seeds:
            frame = subsets[(subsets.model == model) & (subsets.seed == seed)]
            require(set(frame.subset) == set(ALL_SUBSET_KEYS), "Modality subset set differs")
            for key in ALL_SUBSET_KEYS:
                part = frame[frame.subset == key]
                require(set(part.case_id) == robust_ids and part.group_id.nunique() == 51,
                        "Robustness representatives differ")
                subset_seed_rows.append({"model": model, "seed": seed, "subset": key,
                                         "group_macro_dice": part.macro_dice.mean(), "n_groups": 51})
            part = shapley[(shapley.model == model) & (shapley.seed == seed)].set_index("case_id")
            require(set(part.index) == robust_ids and np.isfinite(part[list(MODALITIES)]).all().all(),
                    "Shapley cohort or values differ")
            full = frame[frame.subset == subset_key(MODALITIES)].set_index("case_id").macro_dice
            close(part.full_value, full.loc[part.index], "Shapley full-modality value differs")
            close(part[list(MODALITIES)].sum(axis=1), part.full_value, "Shapley efficiency differs")
            for modality in MODALITIES:
                shapley_seed_rows.append({"model": model, "seed": seed, "modality": modality,
                                          "mean_contribution": part[modality].mean(), "n_groups": 51})
        runs = ledger[ledger.name == model]
        require(set(runs.seed) == set(seeds) and runs.qualitative_cases.sum() == 3,
                "Evaluation cost matrix differs")
        for field in ["attempt_wall_seconds", "qualitative_wall_seconds", "peak_gpu_gb"]:
            require(np.isfinite(runs[field]).all() and runs[field].ge(0).all(), "Invalid cost")
        require(runs.attempt_wall_seconds.ge(runs.qualitative_wall_seconds).all(), "Qualitative time exceeds total")
        costs.append({"model": model, "n_training_seeds": 3,
                      "evaluation_attempt_seconds_median": runs.attempt_wall_seconds.median(),
                      "evaluation_attempt_seconds_sum": runs.attempt_wall_seconds.sum(),
                      "qualitative_seconds_sum": runs.qualitative_wall_seconds.sum(),
                      "qualitative_case_inferences": int(runs.qualitative_cases.sum()),
                      "peak_allocated_gpu_gib_max": runs.peak_gpu_gb.max()})
    subset_seeds, shapley_seeds = pd.DataFrame(subset_seed_rows), pd.DataFrame(shapley_seed_rows)
    subset_summary = subset_seeds.groupby(["model", "subset"], sort=False).group_macro_dice.agg(
        mean="mean", seed_sd="std").reset_index().assign(n_groups=51, n_training_seeds=3)
    shapley_summary = shapley_seeds.groupby(["model", "modality"], sort=False).mean_contribution.agg(
        mean="mean", seed_sd="std").reset_index().assign(n_groups=51, n_training_seeds=3, empty_value=0.0)
    session = read(source / "session_status.json")
    require(session["all_planned_completed"] and not session["pending"] and not session["failures"],
            "Session is incomplete")
    dest.mkdir(parents=True, exist_ok=True)
    tables = {"regional_results_by_seed.csv": regions, "regional_summary.csv": regional_summary,
              "modality_subset_by_seed.csv": subset_seeds, "modality_subset_summary.csv": subset_summary,
              "modality_shapley_by_seed.csv": shapley_seeds, "modality_shapley_summary.csv": shapley_summary,
              "resource_summary.csv": pd.DataFrame(costs)}
    for name, frame in tables.items():
        frame.to_csv(dest / name, index=False)
    require(inputs == {name: sha(source / name) for name in input_names}, "Source changed during reporting")
    provenance = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "report_complete": True,
                  "models": models, "seeds": seeds, "input_hashes": inputs,
                  "output_hashes": {name: sha(dest / name) for name in tables},
                  "report_source_sha256": sha(__file__),
                  "regional_helper_sha256": sha(Path(__file__).with_name("report_segmentation_study.py")),
                  "session_wall_seconds": session["wall_seconds"],
                  "regional_scope": "Nonempty-reference equal-group Dice; primary case macro uses different denominators.",
                  "subset_scope": "Each of all 15 fixed subsets on 51 representatives; mean and sample SD across three fixed seeds.",
                  "shapley_scope": "Macro-Dice attribution across all subsets with a declared zero empty-set value; not causal scanner utility or patient-level validation.",
                  "cost_scope": "Attempt timers include quantitative and qualitative inference; workers overlap. Excludes setup before timers and does not measure isolated latency or GPU busy time.",
                  "raw_images_read": False, "new_training_or_inference": False, "new_hypothesis_tests": False}
    (dest / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"report_complete": True, "tables": {k: len(v) for k, v in tables.items()},
                      "session_wall_seconds": session["wall_seconds"]}, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    describe(args.source, args.output)
