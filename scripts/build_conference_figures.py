#!/usr/bin/env python3
"""Create 20 traceable figures from frozen Kaggle evidence; never train a model."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import itertools
import json
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SEG = ["qmmf", "no_quality", "shuffled_quality", "matched_moment_fusion",
       "no_variance", "no_max", "no_consistency", "hemis", "unet25d"]
PROTECTED = ["qmmf", "no_quality", "hemis", "unet25d"]
CLASS = ["full", "no_attention", "no_multiscale", "no_attention_no_multiscale",
         "no_augmentation", "narrow_resnet_gn", "plain_cnn"]
LABEL = {"qmmf": "QMMF", "no_quality": "No quality", "shuffled_quality": "Shuffled quality",
         "matched_moment_fusion": "Matched moment fusion", "no_variance": "No variance",
         "no_max": "No max", "no_consistency": "No consistency", "hemis": "HeMIS-style",
         "unet25d": "U-Net 2.5D", "full": "Full classifier", "no_attention": "No attention",
         "no_multiscale": "No multiscale", "no_attention_no_multiscale": "Neither module",
         "no_augmentation": "No augmentation", "narrow_resnet_gn": "Narrow ResNet",
         "plain_cnn": "Plain CNN"}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8C6D31",
           "#56B4E9", "#7B61A8", "#68737D", "#BE8700"]
COLOR = dict(zip(SEG, PALETTE))
COLOR.update({m: PALETTE[i] for i, m in enumerate(CLASS)})
COLOR.update(narrow_resnet_gn="#7B61A8", plain_cnn="#56B4E9")
SEEDS = [42, 43, 44]
ENDPOINTS = ["Full modalities", "Mean over subsets", "Worst subset"]


def required_inputs():
    files = [
        "classification/audit/dataset_audit.json", "classification/audit/image_manifest.csv",
        "classification/model_summary.csv", "classification/per_seed_oof_metrics.csv",
        "classification/oof_predictions.csv", "classification/paired_ablation_deltas.csv",
        "segmentation_audit/grouped_v2/case_groups.json", "segmentation_audit/grouped_v2/split_counts.csv",
        "segmentation_main/repeated_seed_metrics.csv", "segmentation_main/seed_specific_metrics.csv",
        "segmentation_main/paired_repeated_seed_deltas.csv", "segmentation_main/all_training_histories.csv",
        "segmentation_main/training_diagnostics.csv", "segmentation_main/resource_summary.csv",
        "segmentation_protected/verification.json", "segmentation_protected/repeated_seed_metrics.csv",
        "segmentation_protected/seed_specific_metrics.csv", "segmentation_protected/paired_repeated_seed_deltas.csv",
        "segmentation_protected/all_full_case_metrics.csv", "segmentation_protected/reporting/regional_summary.csv",
        "segmentation_protected/reporting/modality_subset_summary.csv",
        "segmentation_protected/reporting/modality_shapley_summary.csv",
        "segmentation_protected/qualitative/prespecified_examples.png",
        "segmentation_protected/qualitative/prespecified_examples.pdf",
        "segmentation_protected/qualitative/ATTRIBUTION.md",
        "segmentation_protected/qualitative/panel_metrics.json",
        "segmentation_protected/qualitative/verification.json",
        "segmentation_historical_sensitivity/paired_repeated_seed_deltas.csv",
        "segmentation_historical_sensitivity/verification.json",
        "optimization/kaggle_training_io/timing_comparison.csv",
        "optimization/kaggle_training_io/completion.json",
    ]
    files += [f"segmentation_protected/full_group_macro_dice_seed{s}_by_group.csv" for s in SEEDS]
    return ["results/" + p for p in files]


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#82909B", "axes.labelcolor": "#263743",
        "text.color": "#163449", "xtick.color": "#43535F", "ytick.color": "#43535F",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "mri-conference-v1",
        "lines.linewidth": 1.8, "legend.frameon": False,
    })


class FigureBuilder:
    def __init__(self, root, output):
        self.root = Path(root)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "data").mkdir(exist_ok=True)
        self.used = {}
        self.records = []
        configure_style()

    def source(self, name):
        path = self.root / name
        self.used[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def csv(self, name):
        return pd.read_csv(self.source("results/" + name))

    def js(self, name):
        return json.loads(self.source("results/" + name).read_text())

    def canvas(self, number, title, subtitle, ncols=1, nrows=1, size=(11.6, 5.8), sharey=False):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False, sharey=sharey)
        fig.suptitle(f"{number:02d}  |  {title}", x=.015, y=.99, ha="left", fontsize=17, fontweight="bold")
        fig.text(.015, .935, subtitle, va="top", fontsize=10, color="#536775")
        return fig, axes.ravel()

    def save(self, number, slug, title, fig, data, caption, interpretation, placement, footnote, layout=True):
        stem = f"fig{number:02d}_{slug}"
        if layout:
            fig.tight_layout(rect=(.005, .065, .995, .87), h_pad=2.3, w_pad=2.0)
        fig.text(.015, .015, footnote, fontsize=8.5, color="#526471", va="bottom")
        for ext in ["png", "pdf", "svg"]:
            kwargs = {"dpi": 240} if ext == "png" else {}
            metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
            fig.savefig(self.output / f"{stem}.{ext}", metadata=metadata, **kwargs)
        plt.close(fig)
        self.record(number, stem, title, data, caption, interpretation, placement)

    def record(self, number, stem, title, data, caption, interpretation, placement):
        data.to_csv(self.output / "data" / (stem + ".csv"), index=False)
        outputs = [f"{stem}.{ext}" for ext in ["png", "pdf", "svg"]] + [f"data/{stem}.csv"]
        self.records.append({"number": number, "stem": stem, "title": title, "caption": caption,
                             "interpretation": interpretation, "suggested_placement": placement,
                             "sources": dict(sorted(self.used.items())), "plotted_rows": len(data),
                             "outputs": {p: hashlib.sha256((self.output / p).read_bytes()).hexdigest() for p in outputs}})
        self.used = {}
        print(f"Verified figure {number:02d}: {title}", flush=True)

    def dataset_audit(self):
        c = self.js("classification/audit/dataset_audit.json")
        images = self.csv("classification/audit/image_manifest.csv")
        groups = self.js("segmentation_audit/grouped_v2/case_groups.json")["case_to_group"]
        assert (c["raw_files"], c["images"], c["similarity_groups"]) == (506, 228, 209)
        assert len(images) == c["images"] and images.group_id.nunique() == c["similarity_groups"]
        assert (len(groups), len(set(groups.values()))) == (484, 262)
        rows = [("Classification", "Files", c["raw_files"]), ("Classification", "Unique images", c["images"]),
                ("Classification", "Similarity groups", c["similarity_groups"]),
                ("Segmentation", "Canonical case IDs", len(groups)), ("Segmentation", "Similarity groups", len(set(groups.values())))]
        d = pd.DataFrame(rows, columns=["study", "unit", "count"])
        fig, axes = self.canvas(1, "Audit the sampling unit before measuring performance",
                                "File counts and case IDs overstate the number of independent image groups", ncols=2)
        for ax, study in zip(axes, ["Classification", "Segmentation"]):
            v = d[d.study == study]
            bars = ax.bar(v.unit, v["count"], color=["#B8C9D3", "#0072B2", "#009E73"][:len(v)], width=.58)
            ax.bar_label(bars, fontsize=14, padding=6); ax.set_ylim(0, 580); ax.set_ylabel("Count")
            ax.set_title(study, loc="left"); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        self.save(1, "dataset_audit", "Dataset audit", fig, d,
                  "Classification auditing reduced 506 files to 228 decoded unique images in 209 conservative similarity groups. The segmentation dataset contains 484 canonical case IDs in 262 conservative groups after the rescaled-copy audit. Counts describe different units and must not be interpreted as patient counts.",
                  "Image grouping addresses observed duplication; it cannot establish patient independence.", "Methods / data audit",
                  "Source: completed dataset audits. Similarity groups are not verified patients.")

    def cohorts(self):
        d = self.csv("segmentation_audit/grouped_v2/split_counts.csv").query("fold == 0").copy()
        v = self.js("segmentation_protected/verification.json")
        d = pd.concat([d, pd.DataFrame([{"fold": 0, "role": "reserved", "cases": v["full_cases"], "groups": v["full_groups"]}])], ignore_index=True)
        assert d.cases.sum() == 484 and d.groups.sum() == 262
        fig, axes = self.canvas(2, "Frozen segmentation cohorts", "Only development fold 0 was trained; three seeds repeat the same partition")
        ax = axes[0]; x = np.arange(4)
        for shift, key, color in [(-.19, "cases", "#0072B2"), (.19, "groups", "#009E73")]:
            bars = ax.bar(x + shift, d[key], .34, label=key.capitalize(), color=color)
            ax.bar_label(bars, padding=4)
        ax.set_xticks(x, ["Training", "Inner selection", "Outer development", "Reserved evaluation"])
        ax.set_ylabel("Count"); ax.set_ylim(0, 350); ax.legend(ncol=2); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        self.save(2, "segmentation_cohorts", "Frozen segmentation cohorts", fig, d,
                  "The executed fold-0 partition contains 294 training cases/132 groups, 33 inner-selection cases/24 groups, 91 outer-development cases/55 groups, and 66 reserved cases/51 groups. All 27 main fits share this partition. Missing-modality development evaluation uses 16 fixed group representatives; reserved subset evaluation uses 51 representatives. The 15-group historical sensitivity is nested within the reserved set.",
                  "Four-fold metadata exists, but the reported main experiment is one grouped development fold repeated across three seeds.", "Methods / evaluation design",
                  "Historical exposure remains: seven reserved groups contain earlier notebook validation cases.")

    def classification_comparison(self):
        d = self.csv("classification/model_summary.csv").set_index("model").loc[CLASS].reset_index()
        fig, axes = self.canvas(3, "Classification ablations across three seeds", "Seven variants × five grouped folds × three seeds = 105 completed fits", ncols=2, sharey=True)
        for ax, metric, title in zip(axes, ["macro_f1", "auroc"], ["OOF macro F1", "OOF AUROC"]):
            for i, row in d.iterrows():
                ax.errorbar(row[metric + "_mean"], i, xerr=row[metric + "_std"], fmt="o", capsize=3, color=COLOR[row.model])
            ax.set_yticks(range(len(d)), [LABEL[m] for m in d.model]); ax.set_xlabel(title); ax.set_xlim(.45, 1)
            ax.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis()
        self.save(3, "classification_ablations", "Classification ablation summary", fig, d,
                  "Out-of-fold macro F1 and AUROC are summarized over the three fixed seeds for every classifier variant. Points show the seed mean and bars show one sample standard deviation across seeds. Each seed contributes one out-of-fold prediction per unique image. These bars are not confidence intervals.",
                  "The full classifier has mean macro F1 0.7745; its paired advantage over the narrow ResNet remains uncertain.", "Supplement / separate classification benchmark",
                  "228 unique images; 209 conservative groups. Bars: ±1 seed SD, not patient uncertainty.")

    def classification_curves(self):
        oof = self.csv("classification/oof_predictions.csv")
        metrics = self.csv("classification/per_seed_oof_metrics.csv")
        fig, axes = self.canvas(4, "Out-of-fold discrimination", "Full classifier and narrow ResNet; a separate curve for each fixed seed", ncols=2)
        rows = []
        for model in ["full", "narrow_resnet_gn"]:
            for seed, style in zip(SEEDS, ["-", "--", ":"]):
                f = oof[(oof.model == model) & (oof.seed == seed)]
                assert len(f) == 228 and f.image_id.nunique() == 228
                fpr, tpr, threshold = roc_curve(f.y, f.probability_yes)
                precision, recall, pr_threshold = precision_recall_curve(f.y, f.probability_yes)
                metric = metrics[(metrics.model == model) & (metrics.seed == seed)].iloc[0]
                axes[0].plot(fpr, tpr, style, color=COLOR[model], alpha=.85,
                             label=f"{LABEL[model]}, {seed}: {metric.auroc:.3f}")
                axes[1].plot(recall, precision, style, color=COLOR[model], alpha=.85,
                             label=f"{LABEL[model]}, {seed}: {metric.average_precision:.3f}")
                rows += [{"model": model, "seed": seed, "curve": "ROC", "x": x, "y": y} for x, y in zip(fpr, tpr)]
                rows += [{"model": model, "seed": seed, "curve": "PR", "x": x, "y": y} for x, y in zip(recall, precision)]
        axes[0].plot([0, 1], [0, 1], color="#A0ADB6", lw=1, zorder=0)
        prevalence = oof.query("model == 'full' and seed == 42").y.mean()
        axes[1].axhline(prevalence, color="#A0ADB6", lw=1, label=f"Class prevalence: {prevalence:.3f}")
        for ax, xlabel, ylabel, title in zip(axes, ["False-positive rate", "Recall"], ["True-positive rate", "Precision"], ["ROC (legend: AUROC)", "Precision–recall (legend: AP)"]):
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02), title=title)
            ax.legend(fontsize=7.7, loc="lower right" if xlabel == "False-positive rate" else "lower left")
        self.save(4, "classification_roc_pr", "Classification ROC and precision–recall", fig, pd.DataFrame(rows),
                  "ROC and precision–recall curves use the original out-of-fold probabilities for each of three seeds. Each curve contains 228 unique images; repeated seeds are displayed separately and are not pooled as independent observations. The ROC diagonal and positive-class prevalence are reference lines. Legend values are the saved AUROC and average precision.",
                  "These are internal, descriptive image-level discrimination curves, not calibrated clinical operating points.", "Supplement / classifier discrimination",
                  "No new training, threshold selection or confidence band. Five grouped folds contribute to each seed curve.")

    def classification_confusion(self):
        d = self.csv("classification/oof_predictions.csv").query("model == 'full'")
        fig, axes = self.canvas(5, "Classification errors at the frozen 0.5 threshold", "Counts are shown independently for each seed; positive label = tumor image", ncols=3, size=(11.6, 5.3))
        rows = []
        for ax, seed in zip(axes, SEEDS):
            f = d[d.seed == seed]; cm = confusion_matrix(f.y, f.probability_yes >= .5, labels=[0, 1])
            assert cm.sum() == 228
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=141)
            for i, j in itertools.product(range(2), repeat=2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22, color="white" if cm[i, j] > 70 else "#163449")
                rows.append({"seed": seed, "true_label": i, "predicted_label": j, "count": int(cm[i, j]), "threshold": .5})
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No tumor", "Tumor"], yticklabels=["No tumor", "Tumor"], xlabel="Predicted label", ylabel="Reference label", title=f"Seed {seed}")
        self.save(5, "classification_confusion", "Classification confusion matrices", fig, pd.DataFrame(rows),
                  "Out-of-fold confusion matrices for the full classifier at the prespecified probability threshold of 0.5. Every seed uses the same 228 unique images: 87 negative and 141 positive. The image counts cannot be interpreted as independent patients, and the three matrices must not be summed to inflate sample size.",
                  "Both false negatives and false positives remain visible for all fixed seeds.", "Supplement / classifier error analysis",
                  "Threshold fixed at 0.5; no post hoc operating-point optimization.")

    def classification_contrasts(self):
        d = self.csv("classification/paired_ablation_deltas.csv")
        fig, axes = self.canvas(6, "Paired classification contrasts", "Full classifier minus each control; descriptive group-bootstrap intervals")
        ax = axes[0]
        for i, row in d.iterrows():
            control = row.comparison.split(" - ", 1)[1]
            ax.plot([row.ci_low, row.ci_high], [i, i], color=COLOR[control], lw=2.5)
            ax.plot(row.macro_f1_delta, i, "o", color=COLOR[control])
        ax.axvline(0, color="#526471", lw=1); ax.set_yticks(range(len(d)), [LABEL[s.split(" - ", 1)[1]] for s in d.comparison]); ax.invert_yaxis()
        ax.set_xlabel("OOF macro F1 difference: full − control"); ax.grid(axis="x", alpha=.15)
        self.save(6, "classification_paired_contrasts", "Paired classification contrasts", fig, d,
                  "Saved paired contrasts compare the full classifier with all six controls. Points are the mean seed-level OOF macro-F1 differences; bars are descriptive percentile intervals from group resampling with fixed seeds retained. These follow-up intervals are unadjusted for multiplicity. The narrow-ResNet interval crosses zero.",
                  "Use this as exploratory supporting evidence; do not present all positive intervals as confirmatory discoveries.", "Supplement / classifier ablations",
                  "Group bootstrap; fixed training seeds. Unadjusted descriptive intervals; no new hypothesis tests.")

    def endpoint_summary(self, number, folder, models, full_metric, slug, title, subtitle, placement):
        d = self.csv(folder + "/repeated_seed_metrics.csv")
        seeds = self.csv(folder + "/seed_specific_metrics.csv")
        metrics = [full_metric, "mean_subset_macro_dice", "worst_subset_macro_dice"]
        fig, axes = self.canvas(number, title, subtitle, ncols=3, sharey=True, size=(12.5, 6.4 if len(models) > 4 else 5.5))
        for ax, metric, endpoint in zip(axes, metrics, ENDPOINTS):
            for i, model in enumerate(models):
                row = d[(d.model == model) & (d.metric == metric)].iloc[0]
                vals = seeds[(seeds.model == model) & (seeds.metric == metric)].sort_values("seed").value.to_numpy()
                assert len(vals) == 3 and np.isclose(vals.mean(), row["mean"], atol=1e-12)
                ax.plot([row.ci_low, row.ci_high], [i, i], color=COLOR[model], lw=2.4)
                ax.scatter(vals, np.array([i - .10, i, i + .10]), s=17, color=COLOR[model], marker="|", zorder=3)
                ax.scatter(row["mean"], i, color=COLOR[model], s=38, zorder=4)
            ax.set_yticks(range(len(models)), [LABEL[m] for m in models]); ax.set_xlabel("Group macro Dice"); ax.set_xlim(0, 1); ax.set_title(endpoint)
            ax.grid(axis="x", alpha=.15)
        axes[0].invert_yaxis()
        cohort = "55 outer-development groups (full modalities) and 16 fixed representatives (subsets)" if folder == "segmentation_main" else "51 reserved groups: 66 cases for full modalities and 51 representatives for subsets"
        self.save(number, slug, title, fig, d,
                  f"Model comparison across full-modality, mean-subset and worst-subset Dice. The cohort comprises {cohort}. Filled points show means across the three fixed seeds; small ticks show individual seed values. Horizontal bars show saved 95% group-bootstrap intervals retaining all three seeds. Interval width is group uncertainty, not seed SD. The mean- and worst-subset endpoints use all 15 nonempty modality subsets.",
                  "Quality conditioning does not establish an advantage over the no-quality control; the paired intervals are the appropriate comparisons.", placement,
                  "Bars: 95% group-bootstrap intervals, fixed seeds. Ticks: seeds 42/43/44. Groups are not verified patients.")

    def segmentation_contrasts(self, number, folder, full_metric, models, slug, title, family):
        d = self.csv(folder + "/paired_repeated_seed_deltas.csv")
        fig, axes = self.canvas(number, title, f"QMMF minus control; Bonferroni intervals for {family} controls per endpoint", ncols=3, sharey=True, size=(12.5, 6.3 if len(models) > 4 else 5.3))
        for ax, metric, endpoint in zip(axes, [full_metric, "mean_subset_macro_dice", "worst_subset_macro_dice"], ENDPOINTS):
            for i, model in enumerate(models):
                row = d[(d.control == model) & (d.metric == metric)].iloc[0]
                assert int(row.comparison_family_size) == family
                ax.plot([row.bonferroni_ci_low, row.bonferroni_ci_high], [i, i], color=COLOR[model], lw=2.4)
                ax.plot(row.difference, i, "o", color=COLOR[model])
            ax.axvline(0, color="#526471", lw=1); ax.grid(axis="x", alpha=.15)
            ax.set_yticks(range(len(models)), [LABEL[m] for m in models]); ax.set_title(endpoint); ax.set_xlabel("Dice difference: QMMF − control")
        axes[0].invert_yaxis()
        self.save(number, slug, title, fig, d,
                  f"All declared QMMF-control contrasts for the three segmentation endpoints. Bars are the saved Bonferroni-adjusted paired group-bootstrap intervals, using a family of {family} controls separately for each endpoint and retaining three fixed training seeds. Negative values favor the control. The intervals are reproduced from the frozen analysis; no new comparisons were selected for this figure.",
                  "The main development comparisons do not support the quality module; all reserved no-quality contrasts favor removing it.", "Main paper / ablation evidence" if number == 8 else "Main paper / reserved comparisons",
                  "Paired resampling unit: conservative image group. Adjustment is within endpoint, not across the entire figure gallery.")

    def learning_curves(self):
        h = self.csv("segmentation_main/all_training_histories.csv")
        d = h.dropna(subset=["selection_score"]).copy()
        diag = self.csv("segmentation_main/training_diagnostics.csv")
        assert len(d) == 27 * 12 and len(diag) == 27
        fig, axes = self.canvas(9, "Inner-selection learning curves", "All nine variants and all three seeds; selected epochs marked with circles", ncols=3, nrows=3, size=(12.5, 10.5))
        for ax, model in zip(axes, SEG):
            for seed, color in zip(SEEDS, ["#0072B2", "#D55E00", "#009E73"]):
                f = d[(d.model == model) & (d.seed == seed)].sort_values("epoch")
                selected = diag[(diag.model == model) & (diag.seed == seed)].iloc[0].selected_epoch_one_based
                point = f[f.epoch + 1 == selected]
                assert len(point) == 1
                ax.plot(f.epoch + 1, f.selection_score, color=color, label=str(seed), lw=1.6)
                ax.plot(selected, point.selection_score.iloc[0], "o", color=color, markersize=5)
            ax.set(xlabel="Completed epoch", ylabel="Selection score", xlim=(8, 122), ylim=(.25, .8)); ax.set_title(LABEL[model], loc="left")
            ax.grid(alpha=.15)
        axes[0].legend(ncol=3, fontsize=8, loc="lower right")
        self.save(9, "training_selection_curves", "Training selection curves", fig, d,
                  "Inner-validation selection scores at all 12 scheduled checks for every model and seed. Each fit ran 120 epochs; dots indicate the checkpoint selected by the original inner-validation rule. Lines connect observed checks without smoothing or extrapolation. The vertical score is the recorded composite selection objective, not the outer or reserved test score.",
                  "The fixed budget and available validation trajectory do not establish convergence for every model.", "Supplement / optimization diagnostics",
                  "27 fits × 12 checks = 324 observations. Same 120-epoch budget; no test-based checkpoint selection.")

    def compute(self):
        resource = self.csv("segmentation_main/resource_summary.csv")
        score = self.csv("segmentation_main/repeated_seed_metrics.csv").query("metric == 'outer_macro_dice'")
        d = resource.merge(score[["model", "mean", "seed_sd"]], on="model", validate="one_to_one").set_index("model").loc[SEG].reset_index()
        fig, axes = self.canvas(10, "Accuracy, capacity and recorded compute", "Measured resource use from the 27 main Kaggle fits; all models retained", ncols=3, sharey=True, size=(14, 6.8))
        for i, row in d.iterrows():
            axes[0].barh(i, row.parameters / 1e6, color=COLOR[row.model], height=.6)
            axes[0].text(row.parameters / 1e6 + .025, i, f"{row.parameters / 1e6:.3f}", va="center", fontsize=8)
            axes[1].errorbar(row["mean"], i, xerr=row.seed_sd, fmt="o", color=COLOR[row.model], capsize=3)
        axes[0].set(xlabel="Trainable parameters (millions)", xlim=(0, 1.5))
        axes[1].set(xlabel="Outer Dice (mean ± seed SD)", xlim=(.65, .85))
        for key, label, color in [("fit_and_validation_seconds_median", "Fit + inner validation", "#0072B2"), ("outer_evaluation_seconds_median", "Full evaluation", "#009E73"), ("subset_evaluation_seconds_median", "Subset evaluation", "#D55E00")]:
            bottom = np.zeros(len(d)) if key == "fit_and_validation_seconds_median" else d.fit_and_validation_seconds_median.to_numpy() / 60
            if key == "subset_evaluation_seconds_median":
                bottom += d.outer_evaluation_seconds_median.to_numpy() / 60
            axes[2].barh(np.arange(len(d)), d[key] / 60, left=bottom, color=color, label=label)
        axes[0].set_yticks(range(len(d)), [LABEL[m] for m in d.model]); axes[0].invert_yaxis()
        axes[2].set_xlabel("Component median durations (minutes)")
        axes[2].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=1)
        for ax in axes: ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
        self.save(10, "segmentation_compute", "Segmentation capacity and compute", fig, d,
                  "Aligned rows show trainable parameters (left), mean outer-development Dice with one seed SD (center), and component-wise median fit/validation and evaluation durations over three seeds (right). Duration components are stacked for readability; their sum is not the median total runtime. Cross-family objectives differ: QMMF and its controls have auxiliary heads while HeMIS-style and U-Net do not.",
                  "This is an observed implementation trade-off, not a controlled architecture-only efficiency claim or an account billing estimate.", "Supplement / resources",
                  "Kaggle T4 measurements. Resource medians describe components; auxiliary-objective differences remain a confound.")

    def regions(self):
        d = self.csv("segmentation_protected/reporting/regional_summary.csv")
        fig, axes = self.canvas(13, "Reserved regional segmentation", "Nonempty-reference regional Dice; group aggregation followed by three-seed averaging", ncols=3, sharey=True)
        for ax, region, title in zip(axes, ["wt", "tc", "et"], ["Whole tumor (WT)", "Tumor core (TC)", "Enhancing tumor (ET)"]):
            for i, model in enumerate(PROTECTED):
                row = d[(d.model == model) & (d.region == region)].iloc[0]
                ax.errorbar(row.mean_group_dice_nonempty_reference, i, xerr=row.seed_sd, fmt="o", color=COLOR[model], capsize=3)
            ax.set_yticks(range(4), [LABEL[m] for m in PROTECTED]); ax.set_xlim(0, 1); ax.set_title(title); ax.set_xlabel("Regional Dice"); ax.grid(axis="x", alpha=.15)
        axes[0].invert_yaxis()
        self.save(13, "reserved_regional_dice", "Reserved regional Dice", fig, d,
                  "Group-aggregated Dice for WT, TC and ET in the reserved evaluation. Points show the mean over three seeds and bars show seed SD. The frozen nonempty-reference rule excludes two ET-empty cases from ET Dice; WT and TC have 66 nonempty reference cases, while ET has 64. False-positive ET on the excluded cases is reported separately in Figure 17.",
                  "Regional averages must be read together with the empty-reference failure analysis.", "Main paper / regional results",
                  "Bars: ±1 seed SD. Nonempty reference only; two ET-empty cases are reported separately.")

    def modality_subsets(self):
        d = self.csv("segmentation_protected/reporting/modality_subset_summary.csv")
        channels = ["t1", "t1ce", "t2", "flair"]
        order = ["+".join(c) for n in range(1, 5) for c in itertools.combinations(channels, n)]
        assert set(d.subset) == set(order) and len(d) == 60
        matrix = d.pivot(index="model", columns="subset", values="mean").loc[PROTECTED, order]
        fig, axes = self.canvas(14, "All 15 available-modality combinations", "Mean group Dice over three seeds; each cell represents the same 51 reserved groups", size=(14, 6.5))
        ax = axes[0]; im = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis", aspect="auto")
        for i, j in itertools.product(range(4), range(15)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if matrix.iloc[i, j] < .48 else "#152F41")
        ax.set_yticks(range(4), [LABEL[m] for m in PROTECTED])
        ax.set_xticks(range(15), [s.upper().replace("+", "\n+") for s in order], fontsize=8)
        for boundary in [3.5, 9.5, 13.5]: ax.axvline(boundary, color="white", lw=2)
        fig.colorbar(im, ax=ax, shrink=.8, pad=.02, label="Group macro Dice")
        self.save(14, "reserved_modality_subsets", "All reserved modality subsets", fig, d,
                  "Heatmap of all 15 nonempty subsets of T1, T1ce, T2 and FLAIR for four models. Values are group-level scores averaged over the three fixed seeds and the 51 reserved representatives. The scale is fixed at 0–1 and each value is shown numerically. Complete four-channel preprocessing occurs before channel masking, so the experiment measures simulated missingness.",
                  "No quality has the higher descriptive mean than QMMF in all 15 combinations; individual cells are not new adjusted hypothesis tests.", "Main paper / missing-modality robustness",
                  "Complete-modality preprocessing precedes masking. Cell differences are descriptive; no per-cell significance claims.")

    def shapley(self):
        d = self.csv("segmentation_protected/reporting/modality_shapley_summary.csv")
        fig, axes = self.canvas(15, "Modality contributions under the declared subset game", "Shapley attribution of group Dice; empty-subset utility is fixed at zero")
        ax = axes[0]; modalities = ["t1", "t1ce", "t2", "flair"]; x = np.arange(4)
        for j, model in enumerate(PROTECTED):
            f = d[d.model == model].set_index("modality").loc[modalities]
            assert np.allclose(f.empty_value, 0)
            ax.bar(x + (j - 1.5) * .19, f["mean"], .17, yerr=f.seed_sd, capsize=3, color=COLOR[model], label=LABEL[model])
        ax.set_xticks(x, [s.upper() for s in modalities]); ax.set_ylabel("Shapley contribution to Dice"); ax.set_ylim(0, .4); ax.legend(ncol=2)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        self.save(15, "reserved_modality_shapley", "Reserved modality Shapley contributions", fig, d,
                  "Shapley values are computed from all 15 observed nonempty subset scores and a declared empty-subset value of zero. Group means are averaged across the three seeds; bars show seed SD. Contributions are conditional on the fitted models, the preprocessing and the chosen utility convention, and are not causal importance estimates.",
                  "T1ce has the largest descriptive contribution in this declared game; no claim about clinical indispensability follows.", "Supplement / modality attribution",
                  "Empty utility = 0; fixed subset game. Bars: ±1 seed SD. Attribution is descriptive, not causal.")

    def group_heterogeneity(self):
        frames = []
        for seed in SEEDS:
            f = self.csv(f"segmentation_protected/full_group_macro_dice_seed{seed}_by_group.csv").set_index("group_id")
            frames.append(f[PROTECTED])
        d = pd.concat(frames).groupby(level=0).mean().reset_index()
        assert len(d) == 51
        d["qmmf_minus_no_quality"] = d.qmmf - d.no_quality
        fig, axes = self.canvas(16, "Group-level heterogeneity behind the average", "Reserved full-modality Dice, averaged over the same three fixed seeds", ncols=2)
        axes[0].scatter(d.no_quality, d.qmmf, s=35, color="#0072B2", alpha=.8)
        axes[0].plot([0, 1], [0, 1], color="#8F9DA7", lw=1)
        axes[0].set(xlabel="No-quality control", ylabel="QMMF", xlim=(0, 1), ylim=(0, 1)); axes[0].set_aspect("equal")
        axes[1].hist(d.qmmf_minus_no_quality, bins=np.linspace(-.10, .10, 17), color="#0072B2", edgecolor="white")
        assert d.qmmf_minus_no_quality.between(-.10, .10).all(), "Histogram range would omit groups"
        axes[1].axvline(0, color="#526471", lw=1); axes[1].axvline(d.qmmf_minus_no_quality.mean(), color="#D55E00", ls="--", label=f"Mean: {d.qmmf_minus_no_quality.mean():+.4f}")
        axes[1].set(xlabel="Group Dice difference: QMMF − no quality", ylabel="Number of groups"); axes[1].legend()
        self.save(16, "reserved_group_heterogeneity", "Reserved group heterogeneity", fig, d,
                  "Each scatter point is one of 51 conservative reserved image groups after averaging the three fixed seeds. The diagonal denotes equal performance. The histogram displays every corresponding paired group difference and its mean, rather than selecting only favorable or unfavorable examples. It adds no hypothesis test.",
                  "The aggregate negative difference coexists with group-specific gains and losses; neither plot establishes patient-level effects.", "Supplement / distribution of errors",
                  "51 group means; each group appears once. No pooled-seed sample inflation or post hoc group selection.")

    def empty_reference(self):
        full = self.csv("segmentation_protected/all_full_case_metrics.csv")
        d = full[(full.region == "et") & full.reference_empty.astype(str).str.lower().eq("true")].copy()
        assert len(d) == 24 and set(d.case_id) == {"BRATS_023", "BRATS_027"}
        assert (d.volume_ref_ml == 0).all() and (d.volume_pred_ml > 0).all()
        fig, axes = self.canvas(17, "False-positive enhancing tumor on empty references", "Every model and seed predicts ET in both reference-empty reserved cases", ncols=2, sharey=True)
        for ax, case in zip(axes, ["BRATS_023", "BRATS_027"]):
            for i, model in enumerate(PROTECTED):
                f = d[(d.case_id == case) & (d.model == model)].sort_values("seed")
                for j, marker in enumerate(["o", "s", "^"]):
                    ax.scatter(f.volume_pred_ml.iloc[j], i + (j - 1) * .12, marker=marker, s=40, color=COLOR[model])
            ax.set_yticks(range(4), [LABEL[m] for m in PROTECTED]); ax.set_xlim(0, d.volume_pred_ml.max() * 1.2)
            ax.set_xlabel("False-positive ET volume (mL)"); ax.set_title(case); ax.grid(axis="x", alpha=.15)
        axes[0].invert_yaxis()
        axes[1].legend(handles=[Line2D([0], [0], marker=m, color="#68737D", ls="", label=f"Seed {s}") for s, m in zip(SEEDS, ["o", "s", "^"])], fontsize=8, loc="lower right")
        self.save(17, "empty_reference_et_failures", "Empty-reference ET failures", fig, d,
                  "False-positive ET volume for both reserved cases whose reference ET volume is zero. All four models and all three seeds are shown (24 observations). Every observation has positive predicted ET volume. These cases are excluded from the primary nonempty-reference ET Dice according to the frozen rule, but their failures remain explicitly reported here.",
                  "The aggregate Dice results do not imply reliable rejection of absent enhancing tumor.", "Main paper / failure analysis",
                  "Reference ET = 0 mL in both cases. Volumes are measured outputs, not synthetic examples.")

    def historical(self):
        full = self.csv("segmentation_protected/paired_repeated_seed_deltas.csv")
        subset = self.csv("segmentation_historical_sensitivity/paired_repeated_seed_deltas.csv")
        verify = self.js("segmentation_historical_sensitivity/verification.json")
        assert verify["is_exploratory"] and verify["full_groups"] == 15
        d = pd.concat([full.assign(cohort="All 51 reserved groups"), subset.assign(cohort="15-group historical sensitivity")], ignore_index=True).query("control == 'no_quality'")
        fig, axes = self.canvas(18, "Historical exposure sensitivity", "QMMF minus no quality; retain the complete cohort and the smaller exploratory subset", ncols=3, sharey=True, size=(12.5, 5.4))
        labels = ["All 51 reserved groups", "15-group sensitivity"]
        for ax, metric, endpoint in zip(axes, ["full_group_macro_dice", "mean_subset_macro_dice", "worst_subset_macro_dice"], ENDPOINTS):
            f = d[d.metric == metric]
            for i, (_, row) in enumerate(f.iterrows()):
                ax.plot([row.bonferroni_ci_low, row.bonferroni_ci_high], [i, i], color=["#0072B2", "#D55E00"][i], lw=2.5)
                ax.plot(row.difference, i, "o", color=["#0072B2", "#D55E00"][i])
            ax.axvline(0, color="#526471", lw=1); ax.set_yticks([0, 1], labels); ax.set_ylim(1.6, -.6); ax.set_title(endpoint)
            ax.set_xlabel("Dice difference: QMMF − no quality"); ax.grid(axis="x", alpha=.15)
        self.save(18, "historical_exposure_sensitivity", "Historical exposure sensitivity", fig, d,
                  "QMMF minus no-quality differences for the complete 51-group reserved cohort and the frozen 15-group/18-case sensitivity whose groups are entirely outside the original notebook's development partition. Bars reproduce adjusted intervals (three controls per endpoint). The sensitivity was frozen before reserved outcomes but after some main development results were available; it remains exploratory.",
                  "The smaller sensitivity leaves full and mean-subset differences uncertain while still favoring no quality on worst-subset Dice. It does not replace the complete reserved report.", "Main paper / limitations and sensitivity",
                  "The two cohorts overlap; their intervals are not an independent between-cohort test.")

    def qualitative(self):
        base = "results/segmentation_protected/qualitative/"
        data = pd.DataFrame(self.js("segmentation_protected/qualitative/panel_metrics.json"))
        verification = self.js("segmentation_protected/qualitative/verification.json")
        assert len(data) == 12 and set(data.seed) == {42}
        stem = "fig19_prespecified_qualitative"
        for ext in ["png", "pdf"]:
            shutil.copy2(self.source(base + "prespecified_examples." + ext), self.output / (stem + "." + ext))
        from PIL import Image
        png = (self.output / (stem + ".png")).read_bytes()
        with Image.open(self.output / (stem + ".png")) as image:
            width, height = image.size
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>Prespecified MRI prediction examples; CC BY-SA 4.0</title><image width="{width}" height="{height}" xlink:href="data:image/png;base64,{base64.b64encode(png).decode()}"/></svg>'
        (self.output / (stem + ".svg")).write_text(svg)
        attribution = self.source(base + "ATTRIBUTION.md")
        shutil.copy2(attribution, self.output / "ATTRIBUTION.md")
        self.record(19, stem, "Prespecified qualitative predictions", data,
                    "Unmodified, verified prediction montage for the three cases fixed before reserved evaluation: BRATS_143 (slice 77), BRATS_119 (slice 87) and BRATS_023 (slice 77), with all four models at seed 42. Titles report whole-volume macro Dice, not slice Dice. BRATS_143 appeared in the original notebook's validation subset. The figure contains real Kaggle predictions and is reused byte-for-byte in PNG and PDF; the SVG is a raster wrapper.",
                    "Illustrative examples are prespecified and retain failures and historical exposure. They do not estimate population performance.", "Main paper / qualitative examples")

    def throughput(self):
        d = self.csv("optimization/kaggle_training_io/timing_comparison.csv")
        completion = self.js("optimization/kaggle_training_io/completion.json")
        assert completion["completed"] and len(d) == 4
        fig, axes = self.canvas(20, "Lossless cache optimization on Kaggle", "Same model and timed workload; 10 warmup + 100 measured microbatches per format", ncols=2)
        models = ["qmmf_net", "hemis25d"]
        for ax, metric, ylabel in zip(axes, ["central_slices_per_second", "seconds_per_microbatch"], ["Central training windows per second", "Seconds per microbatch"]):
            for j, (cache, label, color) in enumerate([("volume", "Original volume cache", "#A3B8C6"), ("slab8", "Lossless slab cache", "#0072B2")]):
                f = d[d.training_cache == cache].set_index("model").loc[models]
                bars = ax.bar(np.arange(2) + (j - .5) * .34, f[metric], .3, color=color, label=label)
                ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
            ax.set_xticks([0, 1], ["QMMF", "HeMIS-style"]); ax.set_ylabel(ylabel); ax.set_ylim(0, d[metric].max() * 1.28)
            ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        axes[0].legend(fontsize=8)
        self.save(20, "cache_throughput", "Lossless cache throughput", fig, d,
                  "Measured training-segment throughput and latency for original volume caches versus verified voxel-exact slab caches. Each model/format used 10 warmup and 100 timed microbatches with batch size four and accumulation two. The measured throughput ratios are approximately 4.02× for QMMF and 3.51× for HeMIS-style. These are single bounded measurements, not repeated-run confidence estimates.",
                  "The speedups apply to the timed training segment and must not be described as end-to-end experiment speedups.", "Supplement / implementation efficiency",
                  "Bounded training segment only. Cache values were verified lossless; no new model fit or accuracy estimate.")

    def run(self):
        self.dataset_audit(); self.cohorts(); self.classification_comparison(); self.classification_curves()
        self.classification_confusion(); self.classification_contrasts()
        self.endpoint_summary(7, "segmentation_main", SEG, "outer_macro_dice", "main_segmentation_endpoints",
                              "Main segmentation: all nine variants", "27 completed fits; full and missing-modality development endpoints", "Main paper / main ablations")
        self.segmentation_contrasts(8, "segmentation_main", "outer_macro_dice", SEG[1:], "main_paired_ablations", "Main paired ablation intervals", 8)
        self.learning_curves(); self.compute()
        self.endpoint_summary(11, "segmentation_protected", PROTECTED, "full_group_macro_dice", "reserved_segmentation_endpoints",
                              "Reserved evaluation: four selected model families", "12 checkpoint evaluations; 66 cases / 51 conservative groups", "Main paper / primary results")
        self.segmentation_contrasts(12, "segmentation_protected", "full_group_macro_dice", PROTECTED[1:], "reserved_paired_contrasts", "Reserved paired comparisons", 3)
        self.regions(); self.modality_subsets(); self.shapley(); self.group_heterogeneity()
        self.empty_reference(); self.historical(); self.qualitative(); self.throughput()
        assert [r["number"] for r in self.records] == list(range(1, 21))
        inputs = {k: v for r in self.records for k, v in r["sources"].items()}
        assert set(inputs) == set(required_inputs()), (set(inputs) ^ set(required_inputs()))
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "figure_count": 20,
                    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                    "pandas": pd.__version__, "numpy": np.__version__,
                                    "kaggle_working_directory": str(self.root).startswith("/kaggle/working/")},
                    "scope": "Visualization of frozen evidence; no training, inference, retuning or new hypothesis tests.",
                    "figures": self.records}
        (self.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/conference_figures")
    args = parser.parse_args()
    FigureBuilder(args.root, args.output).run()
