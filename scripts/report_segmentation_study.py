#!/usr/bin/env python3
"""Render complete development-study evidence, retaining all seeds and controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze_segmentation import read, require, validate_training_history
from qmmf.config import ExperimentConfig

LABELS = {"qmmf": "QMMF", "hemis": "HeMIS-style", "unet25d": "U-Net 2.5D",
    "no_quality": "No quality", "no_variance": "No variance", "no_max": "No max",
    "no_consistency": "No consistency", "shuffled_quality": "Shuffled quality",
    "matched_moment_fusion": "Capacity-matched\nmoment fusion"}
COLORS = ["#185b7d", "#ca6e2c", "#4e8b4a", "#9a5b9c"]


def export(fig, dest, name):
    for suffix in ["png", "pdf"]:
        fig.savefig(dest / f"{name}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def summarize_regions(frame, models, seeds):
    """Describe verified nonempty-reference regional scores across fixed seeds."""
    keys = ["model", "region", "seed"]
    expected = {(model, region, seed) for model in models for region in ["wt", "tc", "et"] for seed in seeds}
    require(not frame.duplicated(keys).any() and set(map(tuple, frame[keys].to_numpy())) == expected,
            "Regional model/region/seed coverage differs")
    values = frame.group_dice_nonempty_reference.to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
            "Invalid regional Dice")
    cohort_fields = ["n_nonempty_reference", "n_empty_reference", "n_groups_with_nonempty_reference"]
    require(frame.groupby("region")[cohort_fields].nunique().eq(1).all().all(),
            "Regional reference cohorts changed between models or seeds")
    rows = []
    for model in models:
        for region in ["wt", "tc", "et"]:
            data = frame[(frame.model == model) & (frame.region == region)]
            rows.append({"model": model, "region": region,
                "mean_group_dice_nonempty_reference": float(data.group_dice_nonempty_reference.mean()),
                "seed_sd": float(data.group_dice_nonempty_reference.std(ddof=1)),
                "n_training_seeds": len(seeds),
                **{field: int(data[field].iloc[0]) for field in cohort_fields}})
    return pd.DataFrame(rows)


def training_report(sources, dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    rows, histories, regions, inventory, identities = [], [], [], [], set()
    names, scope, split_hash = None, None, None
    for source in map(Path, sources):
        protocol, verification = read(source / "protocol_lock.json"), read(source / "verification.json")
        require(verification["all_planned_completed"] and not verification["locked_test_opened"],
                "Training report requires a complete verified development study")
        if names is not None:
            require(protocol["names"] == names and protocol["scope"] == scope and protocol["split_hash"] == split_hash,
                    "Development studies differ in scope, models, or groups")
        names, scope, split_hash = protocol["names"], protocol["scope"], protocol["split_hash"]
        seed, fold = protocol["seed"], protocol["fold"]
        require((seed,fold) not in identities, "Repeated development seed/fold")
        identities.add((seed,fold))
        region_path = source / "region_and_empty_case_metrics.csv"
        regions.append(pd.read_csv(region_path).assign(seed=seed))
        inventory.append({"path": str(region_path), "sha256": hashlib.sha256(region_path.read_bytes()).hexdigest()})
        for name in names:
            run = source / "runs" / name
            cfg = ExperimentConfig.from_dict(protocol["configs"][name])
            completion = read(run / "completed.json")
            require(cfg.seed == cfg.train.seed == seed and cfg.fold == fold and completion["name"] == name,
                    "Training report identity mismatch")
            history = pd.read_csv(run / "history.csv")
            diagnostics = validate_training_history(history, cfg, completion)
            rows.append({"model": name, "seed": seed, "fold": fold,
                         **completion, **diagnostics})
            histories.append(history.assign(model=name, seed=seed, fold=fold))
            for path in [run / "history.csv", run / "completed.json", source / "protocol_lock.json", source / "verification.json"]:
                inventory.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    require(len({fold for _,fold in identities}) == 1,
            "These learning-curve panels are designed for one development fold")
    require(len(rows) == len(names)*len(identities), "Incomplete training report matrix")
    table, history = pd.DataFrame(rows), pd.concat(histories, ignore_index=True)
    regional = pd.concat(regions, ignore_index=True)
    regional_summary = summarize_regions(regional, names, sorted(table.seed.unique()))
    regional.to_csv(dest / "regional_results_by_seed.csv", index=False)
    regional_summary.to_csv(dest / "regional_summary.csv", index=False)
    table.to_csv(dest / "training_diagnostics.csv", index=False)
    history.to_csv(dest / "all_training_histories.csv", index=False)
    costs = []
    for name in names:
        data = table[table.model == name]
        require(data.parameters.nunique() == 1, "Parameter count changed between training seeds")
        costs.append({"model": name, "parameters": int(data.parameters.iloc[0]),
            "n_training_seeds": data.seed.nunique(),
            "epochs_min": int(data.epochs_run.min()), "epochs_median": float(data.epochs_run.median()),
            "epochs_max": int(data.epochs_run.max()),
            "fit_and_validation_seconds_median": float(data.fit_and_validation_seconds.median()),
            "outer_evaluation_seconds_median": float(data.outer_evaluation_seconds.median()),
            "subset_evaluation_seconds_median": float(data.subset_evaluation_seconds.median()),
            "peak_allocated_gpu_gib_max": float(data.peak_gpu_gb.max()),
            "recorded_fit_validation_evaluation_seconds_sum": float(data.wall_seconds.sum())})
    pd.DataFrame(costs).to_csv(dest / "resource_summary.csv", index=False)
    provenance = {"scope": scope, "seeds": sorted(table.seed.unique().tolist()),
        "models": names, "split_hash": split_hash, "completed_fits": len(table),
        "source_artifacts": inventory, "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_source_sha256": hashlib.sha256((Path(__file__).parent / "analyze_segmentation.py").read_bytes()).hexdigest(),
        "convergence": "Describe budget/early stopping and recent validation changes; no automatic convergence claim.",
        "cost_scope": "Recorded fit/validation/evaluation; excludes preparation and pre-fit quality-normalizer fitting. Runs can overlap in wall time.",
        "optimizer_counts": "Scheduled opportunities; AMP can skip a non-finite-gradient update, so these are not measured successful updates.",
        "regional_summary": "Mean and sample SD across fixed seeds of each verified nonempty-reference equal-group regional Dice. Region denominators differ; these regional means do not reconstruct the case-first primary macro. Empty-reference outcomes remain in regional_results_by_seed.csv.",
        "locked_test_opened": False}
    (dest / "training_diagnostics_provenance.json").write_text(json.dumps(provenance, indent=2))
    curves(dest, history, names, scope)
    return provenance


def curves(dest, history, names, scope):
    nrows = int(np.ceil(len(names)/3))
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 2.8*nrows), squeeze=False)
    seeds = sorted(history.seed.unique())
    for ax, name in zip(axes.flat, names):
        for seed, color in zip(seeds, COLORS):
            frame = history[(history.model == name) & (history.seed == seed)].dropna(subset=["selection_score"])
            ax.plot(frame.epoch+1, frame.selection_score, marker="o", ms=3, lw=1.3,
                    color=color, label=f"Seed {seed}")
            best = frame.loc[frame.selection_score.idxmax()]
            ax.scatter([best.epoch+1], [best.selection_score], marker="*", s=85,
                       color=color, edgecolor="white", linewidth=.5, zorder=4)
        ax.set_title(LABELS.get(name,name), fontsize=10)
        ax.set_ylim(0,1); ax.grid(alpha=.2); ax.set_xlabel("Training epoch")
        ax.set_ylabel("Inner selection score", fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([],[],marker="*",color="gray",linestyle="",markersize=10))
    labels.append("Selected checkpoint")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(f"{scope.title()} development: all fixed seeds, unchanged inner selection rule", fontsize=12)
    fig.tight_layout(rect=[0,.08/nrows,1,.94 if nrows == 1 else .97])
    export(fig,dest,"inner_validation_by_model_seed")


def combined_figures(folder):
    folder = Path(folder)
    receipt = read(folder / "combination_verification.json")
    require(receipt["all_required_completed"], "Combined figures require a complete verified matrix")
    names, seeds = receipt["variants"], receipt["seeds"]
    metrics = pd.read_csv(folder / "repeated_seed_metrics.csv")
    individual = pd.read_csv(folder / "seed_specific_metrics.csv")
    deltas = pd.read_csv(folder / "paired_repeated_seed_deltas.csv")
    endpoints = [("outer_macro_dice", "Full modalities"), ("mean_subset_macro_dice", "Mean over 15 modality subsets")]
    fig, axes = plt.subplots(1,2,figsize=(11,max(3.5,.47*len(names))))
    for ax,(metric,title) in zip(axes,endpoints):
        table = metrics[metrics.metric == metric].set_index("model").loc[names]
        require(len(table) == len(names) and table.index.is_unique, "Repeated combined metric rows")
        y = np.arange(len(names))
        ax.hlines(y, table.ci_low, table.ci_high, color="#153e52", lw=1.6)
        ax.scatter(table["mean"],y,color="#153e52",s=25,zorder=3)
        for index,(seed,color) in enumerate(zip(seeds,COLORS)):
            points = individual[(individual.metric == metric) & (individual.seed == seed)].set_index("model").loc[names]
            ax.scatter(points.value,y+(index-(len(seeds)-1)/2)*.12,color=color,marker="x",s=20,alpha=.8,zorder=4)
        ax.set_yticks(y,[LABELS.get(n,n) for n in names],fontsize=9); ax.invert_yaxis()
        ax.set_xlim(0,1); ax.set_xlabel("Macro Dice"); ax.grid(axis="x",alpha=.2)
        ax.set_title(f"{title}\n{int(table.n_groups.iloc[0])} development groups",fontsize=10)
    handles = [Line2D([],[],marker="o",color="#153e52",label="Mean and descriptive 95% group interval")]
    handles += [Line2D([],[],marker="x",linestyle="",color=c,label=f"Seed {s}") for s,c in zip(seeds,COLORS)]
    fig.legend(handles=handles,loc="lower center",ncol=len(handles),fontsize=8)
    fig.tight_layout(rect=[0,.075,1,1]); export(fig,folder,"repeated_seed_comparison")
    controls = [n for n in names if n != "qmmf"]
    fig, axes = plt.subplots(1,2,figsize=(11,max(3.5,.47*len(controls))))
    for ax,(metric,title) in zip(axes,endpoints):
        table = deltas[deltas.metric == metric].set_index("control").loc[controls]
        require(table.comparison_family_size.eq(len(controls)).all(), "Adjusted comparison family differs")
        y=np.arange(len(controls))
        ax.hlines(y,table.bonferroni_ci_low,table.bonferroni_ci_high,color="#a8b6be",lw=2)
        ax.hlines(y,table.ci_low,table.ci_high,color="#185b7d",lw=4)
        ax.scatter(table.difference,y,color="#153e52",s=24,zorder=3)
        ax.axvline(0,color="#555555",ls="--",lw=.8)
        ax.set_yticks(y,[LABELS.get(n,n) for n in controls],fontsize=9); ax.invert_yaxis()
        ax.set_title(title,fontsize=10); ax.grid(axis="x",alpha=.15)
        ax.set_xlabel("QMMF minus control (macro Dice)")
    fig.legend(handles=[Line2D([],[],color="#185b7d",lw=4,label="Descriptive 95% paired interval"),
        Line2D([],[],color="#a8b6be",lw=2,label="Bonferroni interval across controls per endpoint")],
        loc="lower center",ncol=2,fontsize=8)
    fig.tight_layout(rect=[0,.075,1,1]); export(fig,folder,"paired_ablation_intervals")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("sources",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--combined",type=Path,help="Optional already verified combined result directory")
    a=p.parse_args(); result=training_report(a.sources,a.output)
    if a.combined:
        combined_figures(a.combined)
    print(json.dumps({k:v for k,v in result.items() if k != "source_artifacts"},indent=2))
