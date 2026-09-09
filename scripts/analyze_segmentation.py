#!/usr/bin/env python3
"""Verify and summarize downloaded Kaggle segmentation artifacts, without fitting models.

Bootstrap units are conservative image groups, with identical draws for every model. Intervals are
descriptive development-cohort intervals; they do not cover training randomness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/src"))
from qmmf.config import ExperimentConfig, MODALITIES
from qmmf.shapley import patient_shapley_table
from qmmf.splits import Splits, validate_splits
from qmmf.subsets import ALL_SUBSET_KEYS, subset_key


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=False), message)


def as_bool(series):
    """Reject ambiguous CSV values rather than treating the string 'False' as true."""
    vals = series.astype(str).str.lower()
    require(vals.isin(["true", "false", "0", "1"]).all(), "Invalid empty-region flags")
    return vals.isin(["true", "1"])


def case_macros(frame, expected_cases):
    require(not frame.duplicated(["case_id", "region"]).any(), "Duplicate case/region rows")
    require(set(frame.case_id) == set(expected_cases), "Evaluation cases differ from frozen cohort")
    require(set(frame.region) == {"wt", "tc", "et"}, "Invalid region names")
    require(frame.groupby("case_id").size().eq(3).all(), "A case lacks one of three regions")
    require(np.isfinite(frame.dice).all() and frame.dice.between(0, 1).all(), "Invalid Dice")
    frame = frame.copy()
    frame["reference_empty"] = as_bool(frame.reference_empty)
    frame["prediction_empty"] = as_bool(frame.prediction_empty)
    for col in ["volume_pred_ml", "volume_ref_ml"]:
        require(np.isfinite(frame[col]).all() and frame[col].ge(0).all(), f"Invalid {col}")
    require((frame.reference_empty == frame.volume_ref_ml.eq(0)).all(), "Reference empty flags conflict with volumes")
    require((frame.prediction_empty == frame.volume_pred_ml.eq(0)).all(), "Prediction empty flags conflict with volumes")
    macro = frame[~frame.reference_empty].groupby("case_id").dice.mean().reindex(expected_cases)
    require(np.isfinite(macro).all(), "Case macro Dice is undefined")
    return frame, macro


def aggregate_groups(values, mapping):
    """Independent aggregation of case scores; retain the group index for pairing."""
    require(set(values.index) <= set(mapping), "Missing group assignment")
    require(values.index.is_unique and np.isfinite(values.values).all(), "Invalid case score vector")
    groups = pd.Series([mapping[c] for c in values.index], index=values.index)
    return values.groupby(groups, sort=True).mean()


def validate_training_history(history, cfg, completion):
    """Check schedule/selection independently; describe training without claiming convergence."""
    require(len(history) > 0 and history.epoch.tolist() == list(range(len(history))),
            "Training epochs are missing, repeated, or out of order")
    require(len(history) <= cfg.train.max_epochs, "Training exceeds the frozen epoch budget")
    require(np.isfinite(history.total).all(), "Non-finite training loss history")
    require(np.isfinite(history.epoch_seconds).all() and history.epoch_seconds.gt(0).all(),
            "Invalid training epoch durations")
    require(np.isfinite(history.lr).all() and history.lr.ge(0).all() and
            history.lr.le(cfg.train.lr + 1e-12).all(), "Invalid recorded learning rate")
    selected = history.dropna(subset=["selection_score"])
    expected = [e for e in range(len(history)) if (e+1) % cfg.train.val_every == 0 or e+1 == cfg.train.max_epochs]
    require(selected.epoch.tolist() == expected and len(selected) > 0,
            "Validation history differs from the frozen schedule")
    components = selected[["val_full_macro_dice", "val_mean_subset_macro_dice", "val_worst_subset_macro_dice"]]
    require(np.isfinite(components.values).all() and ((components.values >= 0) & (components.values <= 1)).all(),
            "Invalid inner-validation Dice components")
    recomputed = components.values @ np.array([.6, .3, .1])
    close(selected.selection_score.values, recomputed, "Composite checkpoint-selection score differs")
    best_position = int(np.argmax(recomputed))
    best = selected.iloc[best_position]
    require(int(best.epoch) == completion["best_epoch"], "Selected checkpoint epoch differs from inner history")
    close(best.selection_score, completion["inner_selection_score"], "Selected checkpoint score differs from inner history")
    trailing_checks = len(selected)-1-best_position
    max_epochs_reached = len(history) == cfg.train.max_epochs
    require(max_epochs_reached or (trailing_checks >= cfg.train.early_stopping_patience and
            int(selected.epoch.iloc[-1]) == int(history.epoch.iloc[-1])),
            "A completed fit stopped before either its epoch budget or patience criterion")
    # Check that an earlier patience trigger was not ignored.
    best_so_far, unimproved = -np.inf, 0
    for row in selected.itertuples():
        if row.selection_score > best_so_far:
            best_so_far, unimproved = row.selection_score, 0
        else:
            unimproved += 1
        require(unimproved < cfg.train.early_stopping_patience or row.epoch == history.epoch.iloc[-1],
                "Training continued after the declared early-stopping trigger")
    tail = selected.tail(3)
    return {"epochs_run": len(history), "validation_checks": len(selected),
        "selected_epoch_one_based": int(best.epoch)+1,
        "termination_reason": "epoch_budget_reached" if max_epochs_reached else "validation_patience_reached",
        "best_at_last_validation": best_position == len(selected)-1,
        "last_selection_score": float(selected.selection_score.iloc[-1]),
        "last_minus_best_selection_score": float(selected.selection_score.iloc[-1]-best.selection_score),
        "last_three_validation_gain": float(tail.selection_score.iloc[-1]-tail.selection_score.iloc[0]) if len(tail) == 3 else None,
        "trailing_nonimproving_validation_checks": trailing_checks,
        "scheduled_optimizer_step_opportunities": len(history)*int(np.ceil(cfg.train.steps_per_epoch/cfg.train.accumulation_steps)),
        "central_training_windows_seen": len(history)*cfg.train.steps_per_epoch*cfg.train.batch_size,
        "convergence_interpretation": "descriptive learning-curve diagnostics; no automatic convergence claim"}


def paired_intervals(matrix, draws, seed, metric, unit="case"):
    """matrix: rows=aligned sampling units, columns=models. No fold-as-replicate test."""
    require(len(matrix) > 1 and np.isfinite(matrix.values).all(), "Cannot bootstrap missing or unaligned cases")
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(matrix), size=(draws, len(matrix)))
    boot = matrix.values[ix].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    means = matrix.mean()
    summaries = [{"model": name, "metric": metric, "mean": float(means[name]),
                  "ci_low": float(low[i]), "ci_high": float(high[i]), f"n_{unit}s": len(matrix),
                  "resampling_unit": unit}
                 for i, name in enumerate(matrix.columns)]
    contrasts = []
    if "qmmf" in matrix:
        ref = list(matrix.columns).index("qmmf")
        for i, name in enumerate(matrix.columns):
            if name == "qmmf":
                continue
            delta = boot[:, ref] - boot[:, i]
            lo, hi = np.quantile(delta, [.025, .975])
            contrasts.append({"reference": "qmmf", "control": name, "metric": metric,
                              "difference": float(means.qmmf - means[name]),
                              "ci_low": float(lo), "ci_high": float(hi), f"n_{unit}s": len(matrix),
                              "resampling_unit": unit})
    return summaries, contrasts


def analyze(source, dest, draws=5000):
    source, dest = Path(source), Path(dest)
    protocol = read(source / "protocol_lock.json")
    status = read(source / "session_status.json")
    cohorts = read(source / "development_cohorts.json")
    splits = Splits.load(source / "splits.json")
    validate_splits(splits)
    require(bool(splits.case_to_group), "Case-ID-only split is unsafe for this dataset")
    mapping = splits.case_to_group
    require(protocol["split_hash"] == splits.split_hash, "Protocol split hash mismatch")
    require(not status["locked_test_opened"] and not cohorts["locked_test_opened"], "Unexpected locked-test access")
    require(protocol["test_access"] == "locked_test_closed", "Unexpected protocol test access")
    fold_index = str(protocol["fold"])
    fold = splits.folds[fold_index]
    train, inner, outer, locked = [set(x) for x in [fold["train"], fold["inner_val"],
                                                  cohorts["outer_evaluation"], splits.locked_test]]
    require(not (train & inner or train & outer or inner & outer or (train | inner | outer) & locked),
            "Training, validation, outer, or locked cohorts overlap")
    require(outer == set(fold["outer_val"]), "Outer evaluation differs from the selected fold")
    require(set(cohorts["validation"]) <= inner, "Checkpoint selection used cases outside inner validation")
    require(set(cohorts["robustness"]) <= outer, "Robustness cases are outside outer development fold")
    for key in ["validation", "robustness"]:
        require(len(cohorts[key]) == len({mapping[c] for c in cohorts[key]}),
                f"Representative cohort repeats a similarity group: {key}")
    provenance = read(source / "grouping_provenance.json")
    require(provenance["split_hash"] == splits.split_hash, "Grouping provenance hash mismatch")
    require(not {mapping[c] for c in locked} & set(provenance["pilot_exposed_groups"]),
            "Protected test has a group exposed during the cancelled pilot")
    if protocol["scope"] == "study":
        cache_provenance = read(source / "training_cache_provenance.json")
        require(cache_provenance["completed"] and cache_provenance["voxel_equality_verified_for_all_cases"],
                "Unverified training cache")
        require(protocol["training_read_format"] == f"slab8:{cache_provenance['case_inventory_sha256']}",
                "Training cache provenance mismatch")
    ids, robust_ids = sorted(outer), sorted(cohorts["robustness"])
    names = protocol["names"]
    require(names == status["planned_variants"], "Planned model lists disagree")
    indexed = {}
    for path in (source / "qmmf_runs").rglob("completed.json"):
        record = read(path)
        name = record["name"]
        require(name in names and name not in indexed, f"Unexpected or repeated completed run: {name}")
        indexed[name] = (path.parent, record)
    require(set(indexed) == set(status["completed_variants"]), "Completion ledger differs from run artifacts")
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ["environment.json", "dataset_fingerprint.json", "splits.json", "protocol_lock.json",
                     "session_status.json", "development_cohorts.json", "capacity_match.json", "grouping_provenance.json",
                     "training_cache_provenance.json"]:
        if (source / filename).exists():
            shutil.copy2(source / filename, dest / filename)
    (dest / "runs").mkdir(exist_ok=True)
    macro_matrix, robust_matrix, worst_matrix = {}, {}, {}
    rows, subset_rows, shapley_rows, history_rows, checks, region_rows = [], [], [], [], [], []
    for name in names:
        if name not in indexed:
            continue
        run, completion = indexed[name]
        cfg = ExperimentConfig.from_dict(protocol["configs"][name])
        require(completion["completed"] and not completion["locked_test_opened"], f"Invalid run status: {name}")
        require(completion["config_hash"] == cfg.config_hash(), f"Configuration hash mismatch: {name}")
        require(cfg.data.split_hash == splits.split_hash, f"Configuration split mismatch: {name}")
        require(cfg.fold == int(fold_index) and cfg.seed == cfg.train.seed == protocol["seed"],
                f"Training seed or fold mismatch: {name}")
        norm = read(run / "quality_normalizer.json")
        require(set(norm["training_case_ids"]) == train, f"Normalizer used a different cohort: {name}")
        require(set(norm["training_group_ids"]) == {mapping[c] for c in train},
                f"Normalizer group cohort mismatch: {name}")
        sampling = read(run / "training_sampling.json")
        require(sampling["unit"] == "uniform group then uniform case", "Unexpected training sampling")
        require(sampling["n_groups"] == len({mapping[c] for c in train}) and sampling["n_cases"] == len(train),
                f"Sampling cohort mismatch: {name}")
        donors = sampling["quality_donors"]
        if cfg.ablation == "A17":
            require(set(donors) == train, "Shuffled-quality control is inactive")
        else:
            require(not donors, "Unexpected quality shuffling")
        require(all(donor in train and cid in train and mapping[donor] != mapping[cid]
                    for cid, donor in donors.items()), "Quality donor crosses training roles or stays within a group")
        frame, macros = case_macros(pd.read_csv(run / "outer_case_metrics.csv"), ids)
        summary = read(run / "outer_summary.json")
        require(set(summary["per_case_macro"]) == outer, f"Stored case summary mismatch: {name}")
        close(macros.values, [summary["per_case_macro"][cid] for cid in ids], f"Case macro mismatch: {name}")
        grouped = aggregate_groups(macros, mapping)
        require(set(summary["per_group_macro"]) == set(grouped.index), "Stored group summary mismatch")
        close(grouped.values, [summary["per_group_macro"][g] for g in grouped.index], "Group Dice mismatch")
        close(grouped.mean(), completion["outer_macro_dice"], f"Completion Dice mismatch: {name}")
        close(grouped.mean(), summary["primary_group_macro_dice"], f"Primary group Dice mismatch: {name}")
        close(macros.mean(), completion["outer_case_macro_dice"], f"Secondary case Dice mismatch: {name}")
        require(len(grouped) == completion["outer_n_groups"], "Wrong outer group count")
        close(macros.mean(), summary["summary"]["macro_dice"], f"Summary Dice mismatch: {name}")
        for region, table in frame.groupby("region"):
            nonempty = table[~table.reference_empty]
            value = float(nonempty.dice.mean()) if len(nonempty) else None
            if value is not None:
                close(value, summary["summary"][f"dice_{region}"], f"Region Dice mismatch: {name}/{region}")
            region_rows.append({"model": name, "region": region, "dice_nonempty_reference": value,
                "group_dice_nonempty_reference": float(aggregate_groups(nonempty.set_index("case_id").dice, mapping).mean()) if len(nonempty) else None,
                "n_groups_with_nonempty_reference": len({mapping[c] for c in nonempty.case_id}),
                "n_nonempty_reference": len(nonempty), "n_empty_reference": int(table.reference_empty.sum()),
                "n_empty_predictions": int(table.prediction_empty.sum()),
                "n_empty_reference_false_positives": int((table.reference_empty & ~table.prediction_empty).sum())})
        subset = read(run / "modality_subsets.json")
        require(set(subset["per_case"]) == set(robust_ids), f"Robustness cohort mismatch: {name}")
        for cid, table in subset["per_case"].items():
            require(set(table) == set(ALL_SUBSET_KEYS), f"Missing modality combinations: {name}/{cid}")
            require(np.isfinite(list(table.values())).all(), f"Invalid subset Dice: {name}/{cid}")
            close(table[subset_key(MODALITIES)], macros[cid], f"Repeated full-modality inference differs: {name}/{cid}")
            close(np.mean(list(table.values())), subset["mean_over_subsets"][cid], f"Subset mean mismatch: {name}/{cid}")
            close(np.min(list(table.values())), subset["worst_subset"][cid], f"Worst subset mismatch: {name}/{cid}")
            for key, value in table.items():
                require(0 <= value <= 1, "Subset Dice outside [0, 1]")
                subset_rows.append({"model": name, "case_id": cid, "subset": key,
                                    "group_id": mapping[cid],
                                    "n_modalities": len(key.split("+")), "macro_dice": value})
        shap = patient_shapley_table(subset["per_case"])
        require(shap.efficiency_gap.abs().lt(1e-8).all(), "Shapley efficiency failed")
        for rec in shap.to_dict("records"):
            for modality in MODALITIES:
                close(rec[modality], subset["shapley"][rec["case_id"]][modality], "Stored Shapley differs")
            shapley_rows.append({"model": name, **rec})
        history = pd.read_csv(run / "history.csv")
        diagnostics = validate_training_history(history, cfg, completion)
        macro_matrix[name] = grouped
        robust_matrix[name] = aggregate_groups(pd.Series(subset["mean_over_subsets"]).reindex(robust_ids), mapping)
        worst_matrix[name] = aggregate_groups(pd.Series(subset["worst_subset"]).reindex(robust_ids), mapping)
        close(robust_matrix[name].mean(), completion["robustness_macro_dice"], f"Robustness summary mismatch: {name}")
        rows.append({**completion, **diagnostics, "first_epoch_loss": float(history.total.iloc[0]),
                     "last_epoch_loss": float(history.total.iloc[-1]),
                     "all_predictions_empty": bool(frame.prediction_empty.all()),
                     "train_epoch_seconds": float(history.epoch_seconds.sum())})
        history_rows.extend(history.assign(model=name).to_dict("records"))
        target = dest / "runs" / name
        target.mkdir(exist_ok=True)
        for path in run.glob("*"):
            if path.suffix in {".json", ".csv"}:
                shutil.copy2(path, target / path.name)
        checks.append({"model": name, "case_rows": len(frame), "outer_cases": len(macros),
                       "outer_groups": len(grouped), "group_isolation_and_weighting": "passed",
                       "robustness_cases": len(shap), "subsets_per_case": 15,
                       "split_and_normalizer_isolation": "passed", "metric_recomputation": "passed",
                       "selection_from_inner_only": "passed", "selection_composite_recomputed": "passed",
                       "training_schedule_and_stopping": "passed", "shapley_efficiency": "passed"})
    verification = {"scope": protocol["scope"], "split_hash": splits.split_hash, "checks": checks,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_variants": names, "completed_variants": list(indexed),
        "all_planned_completed": len(indexed) == len(names), "locked_test_opened": False,
        "bootstrap": {"draws": draws, "seed": 20260909, "unit": "conservative_image_similarity_group", "interval": "percentile 95%",
                      "inference": "descriptive, conditional on one training seed and development fold",
                      "multiplicity_adjusted": False},
        "shapley_empty_set_value": 0.0, "shapley_interpretation": "Dice utility attribution, not causal or clinical importance"}
    save(dest / "verification.json", verification)
    if not rows:
        print(json.dumps(verification, indent=2)); return verification
    metrics, contrasts = [], []
    for key, matrix in [("outer_macro_dice", macro_matrix), ("mean_subset_macro_dice", robust_matrix),
                        ("worst_subset_macro_dice", worst_matrix)]:
        frame = pd.DataFrame(matrix)
        frame.to_csv(dest / f"{key}_by_group.csv", index_label="group_id")
        a, b = paired_intervals(frame, draws, 20260909, key, unit="group")
        metrics.extend(a); contrasts.extend(b)
    pd.DataFrame(rows).to_csv(dest / "model_comparison.csv", index=False)
    intervals = pd.DataFrame(metrics)
    intervals.to_csv(dest / "descriptive_group_bootstrap.csv", index=False)
    pd.DataFrame(contrasts).to_csv(dest / "paired_ablation_deltas.csv", index=False)
    pd.DataFrame(region_rows).to_csv(dest / "region_and_empty_case_metrics.csv", index=False)
    subset_frame = pd.DataFrame(subset_rows)
    subset_frame.to_csv(dest / "all_subset_case_metrics.csv", index=False)
    pd.DataFrame(shapley_rows).to_csv(dest / "shapley_by_case.csv", index=False)
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(dest / "training_history.csv", index=False)
    plot_results(dest, intervals, subset_frame, history_frame, [n for n in names if n in indexed])
    inventory = [{"path": str(p.relative_to(source)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(source.rglob("*")) if p.is_file() and p.suffix in {".json", ".csv"}]
    save(dest / "input_artifact_hashes.json", inventory)
    print(pd.DataFrame(rows)[["name", "outer_macro_dice", "robustness_macro_dice", "epochs_run", "wall_seconds"]].to_string(index=False))
    return verification


def plot_results(dest, intervals, subsets, history, names):
    n_robust = subsets.case_id.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, len(names) * .4)))
    for ax, (metric, title) in zip(axes, [("outer_macro_dice", "Full modalities, outer development fold"),
                                          ("mean_subset_macro_dice", f"15-subset mean, {n_robust} development groups")]):
        table = intervals[intervals.metric == metric].set_index("model").loc[names]
        ax.errorbar(table["mean"], range(len(names)),
            xerr=np.maximum(np.vstack([table["mean"] - table.ci_low, table.ci_high - table["mean"]]), 0),
            fmt="o", color="#185b7d", capsize=3)
        ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Macro Dice, descriptive group interval")
        ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(dest / "segmentation_comparison.png", dpi=230)
    fig.savefig(dest / "segmentation_comparison.pdf"); plt.close(fig)
    matrix = subsets.groupby(["model", "subset"]).macro_dice.mean().unstack().reindex(index=names, columns=ALL_SUBSET_KEYS)
    fig, ax = plt.subplots(figsize=(12, max(3, len(names) * .5)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_xticks(range(15), ALL_SUBSET_KEYS, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"Macro Dice by available modalities: fixed {n_robust}-group development sample")
    fig.colorbar(im, ax=ax, label="Macro Dice"); fig.tight_layout()
    fig.savefig(dest / "modality_subset_heatmap.png", dpi=230)
    fig.savefig(dest / "modality_subset_heatmap.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, frame in history.groupby("model", sort=False):
        axes[0].plot(frame.epoch + 1, frame.total, label=name.replace("_", " "))
        validation = frame.dropna(subset=["selection_score"])
        axes[1].plot(validation.epoch + 1, validation.selection_score, marker="o", label=name.replace("_", " "))
    axes[0].set_ylabel("Total training loss (model-specific objective)"); axes[1].set_ylabel("Inner selection score")
    axes[0].set_title("Auxiliary losses differ across model families", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(dest / "learning_curves.png", dpi=230)
    fig.savefig(dest / "learning_curves.pdf"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    analyze(args.source, args.output, args.draws)
