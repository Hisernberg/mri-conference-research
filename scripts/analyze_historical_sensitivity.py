#!/usr/bin/env python3
"""Analyze the frozen historical-cohort subset of verified reserved results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_original_segmentation_exposure import ROOT, derive_exposure, digest, ordered_split, read, require
from combine_segmentation_studies import repeated_seed_summary

PROTOCOL_SHA = "d9de307c680fe7a50183cf3f6ab0752b465b9753be2f33222e2a9d17e355aaf1"


def select_frames(frames, all_groups, selected_groups, models, seeds):
    """Validate the whole source cohort before retaining complete selected groups."""
    require(set(frames) == set(seeds) and len(seeds) == len(set(seeds)), "Training seeds differ")
    require(len(models) == len(set(models)), "Repeated model identity")
    require(len(selected_groups) == len(set(selected_groups)) and len(selected_groups) >= 2,
            "Sensitivity groups must be unique and contain at least two groups")
    require(set(selected_groups) < set(all_groups), "Sensitivity must be a proper cohort subset")
    result = {}
    for seed in seeds:
        frame = frames[seed]
        require(not frame.index.duplicated().any() and not frame.columns.duplicated().any(),
                "Repeated group or model in source")
        require(set(frame.index) == set(all_groups), "Full reserved group coverage differs")
        require(set(frame.columns) == set(models), "Model coverage differs")
        values = frame.to_numpy(dtype=float)
        require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                "Invalid Dice in the full source cohort")
        result[seed] = frame.loc[sorted(selected_groups), models].copy()
    return result


def check_core_completion(receipt, protocol):
    require(receipt.get("all_planned_completed") and receipt.get("completed_checkpoint_evaluations") == 12,
            "Complete core reserved verification is required")
    require(set(receipt["models"]) == set(protocol["models"]) and receipt["seeds"] == protocol["seeds"],
            "Core model or seed identities differ")
    require(receipt["split_hash"] == protocol["current_split_hash"] and
            receipt["main_source_sha256"] == protocol["main_source_sha256"], "Core provenance differs")
    require(receipt["full_cases"] == 66 and receipt["full_groups"] == 51 and
            receipt["robustness_groups"] == 51 and receipt["subsets_per_group"] == 15,
            "Core cohort or subset coverage differs")
    require(receipt["bootstrap_draws"] == protocol["bootstrap_draws"], "Core bootstrap budget differs")
    require(receipt["locked_test_opened"] and not receipt["training_performed"], "Core execution scope differs")
    qualitative = receipt.get("qualitative", {})
    require(qualitative.get("verified") and not qualitative.get("synthetic_fixture", True),
            "Verified nonsynthetic core artifacts are required")


def compare_saved_rows(saved, rows, keys, numeric_fields):
    require(not saved.duplicated(keys).any(), "Repeated core summary key")
    indexed = saved.set_index(keys)
    expected = {tuple(row[k] for k in keys) for row in rows}
    require(set(indexed.index) == expected, "Core summary coverage differs")
    for row in rows:
        actual = indexed.loc[tuple(row[k] for k in keys)]
        require(all(np.isclose(float(actual[k]), float(row[k]), rtol=1e-9, atol=1e-10)
                    for k in numeric_fields), "Core summary does not reproduce from group tables")


def plot_comparison(dest, full, sensitivity, protocol):
    models = protocol["models"]
    labels = {"qmmf": "QMMF", "hemis": "HeMIS", "no_quality": "No quality", "unet25d": "U-Net"}
    titles = ["Full modalities", "Mean of 15 subsets", "Worst subset per case"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, metric, title in zip(axes, protocol["metrics"], titles):
        for data, offset, color, label in [
            (full, -0.12, "#276b9e", "All reserved groups (51)"),
            (sensitivity, 0.12, "#b76916", "Subset of original locked set (15 groups)"),
        ]:
            table = data[data.metric == metric].set_index("model").loc[models]
            errors = np.maximum(np.vstack([table["mean"] - table.ci_low,
                                            table.ci_high - table["mean"]]), 0)
            ax.errorbar(np.arange(len(models)) + offset, table["mean"], yerr=errors,
                        fmt="o", linestyle="none", capsize=2.5, color=color, label=label)
        ax.set_xticks(np.arange(len(models)), [labels[name] for name in models])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Equal-group macro Dice")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, ncol=2, loc="upper center", frameon=False)
    fig.text(0.5, 0.02,
             "Exploratory sensitivity; 95% group intervals retain all three fixed seeds. "
             "Cohorts overlap; differences are descriptive, not causal.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    for suffix in ["png", "pdf"]:
        fig.savefig(dest / ("historical_cohort_sensitivity." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze(source, dest, root=ROOT):
    source, dest, root = Path(source), Path(dest), Path(root)
    protocol_path = root / "audit/historical_exposure_sensitivity_protocol.json"
    require(digest(protocol_path) == PROTOCOL_SHA, "Frozen sensitivity protocol differs")
    protocol = read(protocol_path)
    audit_path = root / "results/segmentation_audit/original_notebook_exposure.json"
    case_path = audit_path.with_suffix(".csv")
    require(digest(audit_path) == protocol["historical_audit_sha256"] and
            digest(case_path) == protocol["historical_case_table_sha256"], "Historical evidence differs")
    base_path = root / "segmentation/configs/grouped_v2/locked_evaluation_protocol.json"
    require(digest(base_path) == protocol["base_quantitative_protocol_sha256"], "Base protocol differs")
    base = read(base_path)
    old_path = root / "results/segmentation_audit/splits.json"
    new_path = source / "splits.json"
    old, new = ordered_split(old_path), ordered_split(new_path)
    require(old["split_hash"] == protocol["original_split_hash"] and
            new["split_hash"] == protocol["current_split_hash"], "Historical or current split differs")
    historical = read(audit_path)
    derived = derive_exposure(old, new, historical["source_evidence"]["validation_case_limit"],
                              historical["source_evidence"]["active_training_folds"])
    require(derived["sensitivity_group_ids"] == protocol["group_ids"] and
            derived["sensitivity_full_case_ids"] == protocol["full_case_ids"], "Sensitivity membership differs")
    groups = set(protocol["group_ids"])
    subset_cases = sorted(c for c in base["robustness_case_ids"] if new["case_to_group"][c] in groups)
    require(subset_cases == protocol["robustness_case_ids"], "Sensitivity representatives differ")
    core_path = source / "verification.json"
    core = read(core_path)
    check_core_completion(core, protocol)
    all_groups = {new["case_to_group"][c] for c in new["locked_test"]}
    require(len(all_groups) == 51 and len(groups) == 15 and len(protocol["full_case_ids"]) == 18,
            "Declared historical cohort counts differ")
    input_paths = {"protocol": protocol_path, "historical_audit": audit_path,
                   "historical_case_table": case_path, "base_protocol": base_path,
                   "old_splits": old_path, "current_splits": new_path, "core_verification": core_path,
                   "core_summary": source / "repeated_seed_metrics.csv",
                   "core_contrasts": source / "paired_repeated_seed_deltas.csv",
                   "analysis_script": Path(__file__),
                   "summary_helper": root / "scripts/combine_segmentation_studies.py"}
    for metric in protocol["metrics"]:
        for seed in protocol["seeds"]:
            name = f"{metric}_seed{seed}_by_group.csv"
            input_paths[name] = source / name
    input_hashes = {name: digest(path) for name, path in input_paths.items()}
    existing = dest / "verification.json"
    if existing.exists():
        receipt = read(existing)
        require(receipt.get("analysis_complete") and receipt["input_hashes"] == input_hashes,
                "Existing sensitivity used different inputs; retain and reconcile it")
        require(all(digest(dest / name) == sha for name, sha in receipt["output_hashes"].items()),
                "Existing sensitivity outputs differ")
        return receipt
    full_rows, full_deltas, rows, deltas, seed_rows = [], [], [], [], []
    selected_tables = {}
    for metric in protocol["metrics"]:
        frames = {seed: pd.read_csv(source / f"{metric}_seed{seed}_by_group.csv", index_col="group_id")
                  for seed in protocol["seeds"]}
        selected = select_frames(frames, all_groups, protocol["group_ids"],
                                 protocol["models"], protocol["seeds"])
        a, b, _ = repeated_seed_summary(frames, metric, draws=protocol["bootstrap_draws"],
                                        seed=protocol["bootstrap_seed"])
        full_rows.extend(a); full_deltas.extend(b)
        a, b, c = repeated_seed_summary(selected, metric, draws=protocol["bootstrap_draws"],
                                       seed=protocol["bootstrap_seed"])
        rows.extend(a); deltas.extend(b); seed_rows.extend(c)
        selected_tables.update({f"{metric}_seed{seed}_by_group.csv": frame for seed, frame in selected.items()})
    compare_saved_rows(pd.read_csv(source / "repeated_seed_metrics.csv"), full_rows, ["model", "metric"],
                       ["mean", "seed_sd", "ci_low", "ci_high", "n_groups", "n_training_seeds"])
    compare_saved_rows(pd.read_csv(source / "paired_repeated_seed_deltas.csv"), full_deltas,
                       ["reference", "control", "metric"],
                       ["difference", "ci_low", "ci_high", "bonferroni_ci_low", "bonferroni_ci_high",
                        "comparison_family_size", "n_groups", "n_training_seeds"])
    require(len(rows) == 12 and len(deltas) == 9 and len(seed_rows) == 36, "Incomplete sensitivity matrix")
    dest.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(dest / "repeated_seed_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(dest / "paired_repeated_seed_deltas.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(dest / "seed_specific_metrics.csv", index=False)
    for name, frame in selected_tables.items():
        frame.to_csv(dest / name, index_label="group_id")
    plot_comparison(dest, pd.DataFrame(full_rows), summary, protocol)
    (dest / "README.md").write_text(
        "# Exploratory historical-cohort sensitivity\n\n"
        "These 15 groups (18 full-modality cases, 15 robustness representatives) belong entirely "
        "to the original locked partition. Selection used recorded metadata before revised reserved "
        "outcomes; it was added after two main development seeds had been reviewed.\n\n"
        "The complete 51-group report remains in results/segmentation_protected. "
        "All four models and three seeds are retained. Intervals resample groups with all fixed "
        "seeds inside each draw; seed SD is separate. The paired table includes unadjusted and "
        "Bonferroni intervals for three controls per endpoint. These overlapping-cohort results "
        "are descriptive, not a causal estimate of historical exposure. The small selected subset "
        "does not establish patient independence, external validation, or absence of undocumented earlier use.\n")
    require(input_hashes == {name: digest(path) for name, path in input_paths.items()},
            "Source changed during sensitivity analysis")
    outputs = {p.name: digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != "verification.json"}
    receipt = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "analysis_complete": True,
        "is_exploratory": True, "protocol_sha256": PROTOCOL_SHA,
        "full_groups": 15, "full_cases": 18, "robustness_groups": 15,
        "models": protocol["models"], "seeds": protocol["seeds"],
        "summary_rows": len(rows), "paired_contrasts": len(deltas), "seed_metric_rows": len(seed_rows),
        "bootstrap_draws": protocol["bootstrap_draws"], "bootstrap_seed": protocol["bootstrap_seed"],
        "full_cohort_report_retained": True, "core_summary_and_contrasts_recomputed": True,
        "raw_images_read": False, "new_training_or_inference": False,
        "input_hashes": input_hashes, "output_hashes": outputs,
    }
    existing.write_text(json.dumps(receipt, indent=2) + "\n")
    print(summary.to_string(index=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
