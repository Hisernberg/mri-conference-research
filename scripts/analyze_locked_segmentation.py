#!/usr/bin/env python3
"""Independently verify protected scores, then pair groups across all fixed seeds."""
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
from analyze_segmentation import read, save, require, close, case_macros, aggregate_groups
from combine_segmentation_studies import repeated_seed_summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
from locked_evaluate import validate_protocol, MAIN_SOURCE_SHA, MODELS, SEEDS
from qmmf.config import ExperimentConfig, MODALITIES
from qmmf.splits import Splits
from qmmf.shapley import patient_shapley_table
from qmmf.subsets import ALL_SUBSET_KEYS, subset_key


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()).hexdigest()


def validate_parent_receipt(provenance, gate):
    seed, name = provenance["seed"], provenance["model"]
    cfg = ExperimentConfig.from_dict(provenance["config"])
    inventory = {item["path"]: item["sha256"] for item in gate["parents"][str(seed)]["artifact_hashes"]}
    stem = f"qmmf_runs/{cfg.run_id()}"
    for rel, data in [(f"configs/{name}.json", provenance["config"]),
                      (f"{stem}/completed.json", provenance["parent_completion"]),
                      (f"{stem}/quality_normalizer.json", provenance["normalizer"])]:
        require(inventory.get(rel) == json_sha(data), "Protected provenance differs from a verified parent artifact")
    parent = provenance["parent_completion"]
    require(parent["completed"] and not parent["locked_test_opened"] and
            parent["best_epoch"] == provenance["selected_epoch"] and parent["name"] == name and
            parent["seed"] == seed, "Protected checkpoint selection differs from its parent")
    close(parent["inner_selection_score"], provenance["inner_selection_score"], "Protected selected score differs")
    require(provenance["frozen_protocol_sha256"] == gate["frozen_protocol_sha256"],
            "Protected checkpoint used a different evaluation protocol")


def verify_one(run, protocol, splits):
    done, provenance = read(run / "completed.json"), read(run / "checkpoint_provenance.json")
    cfg = ExperimentConfig.from_dict(provenance["config"])
    name, seed = done["name"], done["seed"]
    require(done["completed"] and done["locked_test_opened"] and
            done["scope"] == "protected_image_groups", "Invalid protected completion record")
    require(name in MODELS and seed in SEEDS and cfg.seed == cfg.train.seed == seed and cfg.fold == 0,
            "Protected checkpoint identity differs")
    require(provenance["seed"] == seed and provenance["model"] == name, "Wrong checkpoint provenance identity")
    require(not provenance["training_performed"] and not provenance["normalizer_fitted"],
            "Protected cases were used for fitting")
    require(provenance["parent_source_sha256"] == MAIN_SOURCE_SHA, "Wrong protected parent source")
    require(done["parent_config_hash"] == provenance["parent_config_hash"] == cfg.config_hash(),
            "Protected parent configuration hash mismatch")
    require(done["checkpoint_sha256"] == provenance["checkpoint_sha256"] and
            len(done["checkpoint_sha256"]) == 64, "Protected checkpoint hash mismatch")
    require(cfg.data.split_hash == splits.split_hash and cfg.train.max_epochs == 120 and
            cfg.train.steps_per_epoch == 100 and cfg.train.val_every == 10, "Wrong parent training protocol")
    normalizer = provenance["normalizer"]
    train = set(splits.folds["0"]["train"])
    require(set(normalizer["training_case_ids"]) == train and
            set(normalizer["training_group_ids"]) == {splits.case_to_group[c] for c in train},
            "Normalizer crosses the training boundary")
    ids, robust = protocol["full_modality_case_ids"], protocol["robustness_case_ids"]
    frame, macros = case_macros(pd.read_csv(run / "full_case_metrics.csv"), ids)
    summary = read(run / "full_summary.json")
    require(set(summary["per_case_macro"]) == set(ids), "Incomplete stored protected case scores")
    close(macros.values, [summary["per_case_macro"][cid] for cid in ids], "Protected case macro differs")
    groups = aggregate_groups(macros, splits.case_to_group)
    require(set(summary["per_group_macro"]) == set(groups.index), "Protected group summary differs")
    close(groups.values, [summary["per_group_macro"][g] for g in groups.index], "Protected group values differ")
    close(groups.mean(), summary["primary_group_macro_dice"], "Protected primary mean differs")
    close(groups.mean(), done["full_group_macro_dice"], "Protected completion mean differs")
    close(macros.mean(), summary["summary"]["macro_dice"], "Protected case-weighted secondary differs")
    require(done["full_cases"] == len(ids) and done["full_groups"] == len(groups) == 51 and
            done["robustness_groups"] == len(robust) == 51 and done["subsets_per_group"] == 15,
            "Protected completion cohort size differs")
    subset = read(run / "modality_subsets.json")
    require(set(subset["per_case"]) == set(robust) and set(subset["per_case_region"]) == set(robust),
            "Protected subset cases differ")
    rows = []
    for cid in robust:
        table = subset["per_case"][cid]
        require(set(table) == set(ALL_SUBSET_KEYS), "Protected subset combinations incomplete")
        regions = subset["per_case_region"][cid]
        require(set(regions) == {"wt", "tc", "et"}, "Protected subset regions incomplete")
        evaluable = frame[(frame.case_id == cid) & ~frame.reference_empty].region.tolist()
        for region, scores in regions.items():
            require(set(scores) == set(ALL_SUBSET_KEYS), "Protected regional subsets incomplete")
            require(np.isfinite(list(scores.values())).all() and all(0 <= v <= 1 for v in scores.values()),
                    "Invalid protected regional Dice")
        for key, value in table.items():
            close(value, np.mean([regions[r][key] for r in evaluable]), "Protected regional macro differs")
            rows.append({"model": name, "seed": seed, "case_id": cid,
                "group_id": splits.case_to_group[cid], "subset": key, "macro_dice": value})
        close(table[subset_key(MODALITIES)], macros[cid], "Protected repeated full inference differs")
        close(np.mean(list(table.values())), subset["mean_over_subsets"][cid], "Protected subset mean differs")
        close(np.min(list(table.values())), subset["worst_subset"][cid], "Protected worst subset differs")
    robust_groups = aggregate_groups(pd.Series(subset["mean_over_subsets"]), splits.case_to_group)
    worst_groups = aggregate_groups(pd.Series(subset["worst_subset"]), splits.case_to_group)
    close(robust_groups.mean(), done["mean_subset_macro_dice"], "Protected robustness mean differs")
    close(worst_groups.mean(), done["worst_subset_macro_dice"], "Protected worst mean differs")
    shapley = patient_shapley_table(subset["per_case"])
    require(shapley.efficiency_gap.abs().lt(1e-8).all(), "Protected Shapley efficiency failed")
    for row in shapley.to_dict("records"):
        for modality in MODALITIES:
            close(row[modality], subset["shapley"][row["case_id"]][modality], "Protected stored Shapley differs")
    regional = []
    for region, table in frame.groupby("region"):
        nonempty = table[~table.reference_empty].set_index("case_id").dice
        regional.append({"model": name, "seed": seed, "region": region,
            "group_dice_nonempty_reference": aggregate_groups(nonempty, splits.case_to_group).mean(),
            "n_nonempty_reference": len(nonempty), "n_empty_reference": int(table.reference_empty.sum()),
            "n_empty_predictions": int(table.prediction_empty.sum()),
            "n_empty_reference_false_positives": int((table.reference_empty & ~table.prediction_empty).sum())})
    return {"identity": (seed, name), "done": done, "case_metrics": frame.assign(model=name, seed=seed),
        "full_group_macro_dice": groups, "mean_subset_macro_dice": robust_groups,
        "worst_subset_macro_dice": worst_groups, "subsets": rows, "regional": regional,
        "shapley": shapley.assign(model=name, seed=seed)}


def analyze(source, dest, draws=10000):
    source, dest = Path(source), Path(dest)
    frozen, protocol = read(source / "frozen_protocol.json"), read(source / "evaluation_protocol.json")
    splits = Splits.load(source / "splits.json")
    validate_protocol(frozen, splits)
    require(not frozen["locked_test_opened"] and protocol["locked_test_opened"], "Missing protected opening record")
    require(all(protocol[key] == value for key, value in frozen.items() if key != "locked_test_opened"),
            "Execution changed the frozen protected protocol")
    gate = read(source / "development_gate.json")
    receipt = gate["combination_verification"]
    require(gate["main_source_sha256"] == protocol["main_source_sha256"] == MAIN_SOURCE_SHA,
            "Protected parent source receipt differs")
    require(json_sha(frozen) == gate["frozen_protocol_sha256"], "Protected frozen protocol hash differs")
    require(receipt["all_required_completed"] and receipt["completed_model_fits"] == 27 and
            receipt["seeds"] == SEEDS and not receipt["locked_test_opened"],
            "Protected evaluation preceded complete development verification")
    access = read(source / "access_record.json")
    require(access["locked_test_opened"] and access["frozen_protocol_sha256"] == gate["frozen_protocol_sha256"],
            "Protected access protocol receipt differs")
    require(access["development_gate_sha256"] == json_sha(gate),
            "Protected development receipt changed after access")
    require(not protocol["training_performed"] and not protocol["normalizer_fitted"], "Unexpected protected fitting")
    status = read(source / "session_status.json")
    expected = {(s, n) for s in SEEDS for n in MODELS}
    require({(r["seed"], r["name"]) for r in status["planned"]} == expected, "Protected planned matrix differs")
    indexed = {}
    for path in (source / "runs").rglob("completed.json"):
        validate_parent_receipt(read(path.parent / "checkpoint_provenance.json"), gate)
        data = verify_one(path.parent, frozen, splits)
        key = data["identity"]
        require(key not in indexed and key in expected, "Repeated or unexpected protected model/seed")
        indexed[key] = data
    require(set(indexed) == expected and status["all_planned_completed"] and
            {(r["seed"], r["name"]) for r in status["completed"]} == expected,
            "Protected model/seed matrix is incomplete")
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file() and (path.suffix in {".json", ".csv"} or
                (path.suffix == ".png" and "qualitative" in path.parts)) and "subset_progress" not in path.parts:
            target = dest / path.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    summaries, deltas, seed_metrics = [], [], []
    for metric in ["full_group_macro_dice", "mean_subset_macro_dice", "worst_subset_macro_dice"]:
        frames = {s: pd.DataFrame({n: indexed[(s,n)][metric] for n in MODELS}) for s in SEEDS}
        for seed, frame in frames.items():
            frame.to_csv(dest / f"{metric}_seed{seed}_by_group.csv", index_label="group_id")
        a, b, c = repeated_seed_summary(frames, metric, draws=draws)
        summaries.extend(a); deltas.extend(b); seed_metrics.extend(c)
    summary_frame = pd.DataFrame(summaries)
    subset_frame = pd.DataFrame([row for d in indexed.values() for row in d["subsets"]])
    summary_frame.to_csv(dest / "repeated_seed_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(dest / "paired_repeated_seed_deltas.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(dest / "seed_specific_metrics.csv", index=False)
    pd.DataFrame([d["done"] for d in indexed.values()]).to_csv(dest / "all_evaluation_runs.csv", index=False)
    pd.concat([d["case_metrics"] for d in indexed.values()]).to_csv(dest / "all_full_case_metrics.csv", index=False)
    pd.DataFrame([r for d in indexed.values() for r in d["regional"]]).to_csv(dest / "region_and_empty_case_metrics.csv", index=False)
    subset_frame.to_csv(dest / "all_subset_case_metrics.csv", index=False)
    pd.concat([d["shapley"] for d in indexed.values()]).to_csv(dest / "shapley_by_case_seed.csv", index=False)
    verification = {"all_planned_completed": True, "completed_checkpoint_evaluations": len(indexed),
        "models": MODELS, "seeds": SEEDS, "full_cases": 66, "full_groups": 51, "robustness_groups": 51,
        "subsets_per_group": 15, "locked_test_opened": True, "training_performed": False,
        "split_hash": splits.split_hash, "main_source_sha256": MAIN_SOURCE_SHA,
        "bootstrap_draws": draws, "bootstrap_unit": "image group with all fixed seeds retained",
        "multiplicity": "Bonferroni intervals for three QMMF-control comparisons per endpoint",
        "interpretation": "internal reserved image groups; recorded historical validation overlap; restricted eligibility; unverified patient identity",
        "checks": ["frozen cohort and representatives", "all parent fits completed", "training-only normalization",
                   "checkpoint/config identity", "case/regional/group Dice", "15 subsets", "repeated full inference",
                   "Shapley efficiency", "paired group and seed alignment"]}
    if "qualitative_protocol" in gate:
        from analyze_qualitative import verify_and_plot
        verification["qualitative"] = verify_and_plot(source, dest / "qualitative", frozen, splits, gate, indexed)
        verification["checks"].append("prespecified qualitative panels and Dice captions")
    save(dest / "verification.json", verification)
    save(dest / "input_artifact_hashes.json", [{"path": str(p.relative_to(source)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in {".csv", ".json", ".png"}])
    plot(dest, summary_frame, subset_frame)
    print(summary_frame.to_string(index=False))
    return verification


def plot(dest, metrics, subsets):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, (metric, title) in zip(axes, [("full_group_macro_dice", "Full modalities, 51 reserved groups"),
                                         ("mean_subset_macro_dice", "15-subset mean, 51 reserved groups")]):
        table = metrics[metrics.metric == metric].set_index("model").loc[MODELS]
        errors = np.maximum(np.vstack([table["mean"]-table.ci_low, table.ci_high-table["mean"]]), 0)
        ax.errorbar(table["mean"], range(len(MODELS)), xerr=errors, fmt="o", capsize=3, color="#185b7d")
        ax.set_yticks(range(len(MODELS)), [n.replace("_", " ") for n in MODELS]); ax.invert_yaxis()
        ax.set_xlim(0,1); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Mean seed Dice, descriptive group interval")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(dest / f"protected_comparison.{suffix}", dpi=220)
    plt.close(fig)
    matrix = subsets.groupby(["model", "subset"]).macro_dice.mean().unstack().reindex(index=MODELS, columns=ALL_SUBSET_KEYS)
    fig, ax = plt.subplots(figsize=(12,3.6)); im=ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(MODELS)), [n.replace("_", " ") for n in MODELS])
    ax.set_xticks(range(15), ALL_SUBSET_KEYS, rotation=60, ha="right", fontsize=8)
    ax.set_title("Reserved-cohort modality robustness: mean across 51 groups and all three seeds")
    fig.colorbar(im, ax=ax, label="Macro Dice"); fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(dest / f"protected_subset_heatmap.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--draws", type=int, default=10000)
    a=p.parse_args(); analyze(a.source, a.output, a.draws)
