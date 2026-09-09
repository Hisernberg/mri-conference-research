#!/usr/bin/env python3
"""Export manuscript tables without retyping or recomputing experiment results."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
from build_conference_figures import LABEL, SEG, CLASS, PROTECTED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tables"


def markdown(frame):
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def latex(frame):
    def esc(value):
        s = str(value)
        for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]: s = s.replace(a, b)
        return s.replace(" +/- ", " $\\pm$ ")
    rows = ["% Generated table fragment; use the booktabs package.", "\\begin{tabular}{" + "l" * len(frame.columns) + "}", "\\toprule"]
    rows += [" & ".join(esc(x) for x in frame.columns) + " \\\\", "\\midrule"]
    rows += [" & ".join(esc(x) for x in row) + " \\\\" for row in frame.values]
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def build():
    OUT.mkdir(parents=True, exist_ok=True); manifest = []
    def save(name, raw, formatted, sources, note):
        raw.to_csv(OUT / (name + ".csv"), index=False)
        (OUT / (name + ".md")).write_text(markdown(formatted) + "\n" + note + "\n")
        (OUT / (name + ".tex")).write_text(latex(formatted))
        manifest.append({"table": name, "rows": len(formatted), "note": note,
                         "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                         "outputs": {name + "." + ext: hashlib.sha256((OUT / (name + "." + ext)).read_bytes()).hexdigest() for ext in ["csv", "md", "tex"]}})
    def endpoints(name, folder, models, first):
        source = f"results/{folder}/repeated_seed_metrics.csv"; raw = pd.read_csv(ROOT / source)
        rows = []
        for model in models:
            row = {"Model": LABEL[model]}
            for metric, label in [(first, "Full Dice"), ("mean_subset_macro_dice", "Mean-subset Dice"), ("worst_subset_macro_dice", "Worst-subset Dice")]:
                r = raw[(raw.model == model) & (raw.metric == metric)].iloc[0]
                row[label] = f"{r['mean']:.4f} +/- {r.seed_sd:.4f}"
            rows.append(row)
        save(name, raw, pd.DataFrame(rows), [source], "Entries are means +/- sample SD over three fixed training seeds. Group-bootstrap intervals and cohort denominators are in the source CSV and results report.")
    endpoints("main_segmentation", "segmentation_main", SEG, "outer_macro_dice")
    endpoints("reserved_segmentation", "segmentation_protected", PROTECTED, "full_group_macro_dice")
    endpoints("historical_sensitivity", "segmentation_historical_sensitivity", PROTECTED, "full_group_macro_dice")
    source = "results/segmentation_protected/paired_repeated_seed_deltas.csv"; raw = pd.read_csv(ROOT / source); rows = []
    metric_labels = {"full_group_macro_dice": "Full", "mean_subset_macro_dice": "Mean subset", "worst_subset_macro_dice": "Worst subset"}
    for model in PROTECTED[1:]:
        for metric, label in metric_labels.items():
            r = raw[(raw.control == model) & (raw.metric == metric)].iloc[0]
            rows.append({"Control": LABEL[model], "Endpoint": label, "QMMF - control": f"{r.difference:+.4f}", "Adjusted interval": f"[{r.bonferroni_ci_low:+.4f}, {r.bonferroni_ci_high:+.4f}]"})
    save("reserved_contrasts", raw, pd.DataFrame(rows), [source], "10,000 paired group-bootstrap draws retain all three training seeds. Bonferroni adjustment uses three controls separately per endpoint; negative values favor the control.")
    source = "results/classification/model_summary.csv"; raw = pd.read_csv(ROOT / source).set_index("model").loc[CLASS].reset_index()
    rows = [{"Model": LABEL[r.model], "OOF macro F1": f"{r.macro_f1_mean:.4f} +/- {r.macro_f1_std:.4f}", "Accuracy": f"{r.accuracy_mean:.4f}", "AUROC": f"{r.auroc_mean:.4f}"} for _, r in raw.iterrows()]
    save("classification", raw, pd.DataFrame(rows), [source], "Separate image-classification benchmark: 228 unique images, 209 similarity groups, five grouped folds and three seeds. Not a patient-level diagnostic estimate.")
    source = "results/segmentation_protected/reporting/regional_summary.csv"; raw = pd.read_csv(ROOT / source); rows = []
    for model in PROTECTED:
        row = {"Model": LABEL[model]}
        for region in ["wt", "tc", "et"]:
            r = raw[(raw.model == model) & (raw.region == region)].iloc[0]
            row[region.upper()] = f"{r.mean_group_dice_nonempty_reference:.4f} +/- {r.seed_sd:.4f}"
        rows.append(row)
    save("reserved_regions", raw, pd.DataFrame(rows), [source], "Nonempty reference only: WT/TC use 66 cases in 51 groups; ET uses 64 cases in 49 groups. Regional means must not be averaged to reconstruct the case-first headline macro.")
    source = "results/segmentation_main/resource_summary.csv"; raw = pd.read_csv(ROOT / source).set_index("model").loc[SEG].reset_index()
    formatted = pd.DataFrame([{"Model": LABEL[r.model], "Parameters": str(int(r.parameters)), "Median fit + val (min)": f"{r.fit_and_validation_seconds_median/60:.2f}", "Peak allocated (GiB)": f"{r.peak_allocated_gpu_gib_max:.3f}"} for _, r in raw.iterrows()])
    save("resources", raw, formatted, [source], "Observed implementation costs on Kaggle T4. Peak is the maximum across seeds; runtime is the median fit/inner-validation component. Cross-family auxiliary objectives differ.")
    source = "results/segmentation_audit/grouped_v2/split_counts.csv"; raw = pd.read_csv(ROOT / source).query("fold == 0").copy()
    verification = "results/segmentation_protected/verification.json"; v = json.loads((ROOT / verification).read_text())
    raw = pd.concat([raw, pd.DataFrame([{"fold": 0, "role": "reserved", "cases": v["full_cases"], "groups": v["full_groups"]}])], ignore_index=True)
    formatted = raw[["role", "cases", "groups"]].rename(columns={"role": "Role", "cases": "Cases", "groups": "Groups"})
    save("segmentation_cohorts", raw, formatted, [source, verification], "Only fold 0 was executed in the main study. The 15-group historical sensitivity is a nested subset of the reserved cohort.")
    (OUT / "table_manifest.json").write_text(json.dumps({"tables": manifest, "table_count": len(manifest), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    draft = ROOT / "docs/MANUSCRIPT_DRAFT.md"
    if draft.exists():
        text = draft.read_text()
        for row in manifest:
            name = row["table"]
            pattern = r"<!-- BEGIN TABLE:" + name + r" -->.*?<!-- END TABLE:" + name + r" -->"
            replacement = "<!-- BEGIN TABLE:" + name + " -->\n" + (OUT / (name + ".md")).read_text() + "<!-- END TABLE:" + name + " -->"
            text = re.sub(pattern, lambda match: replacement, text, flags=re.S)
        draft.write_text(text)
    print(json.dumps({"tables": len(manifest), "formats": ["csv", "md", "tex"]}))


if __name__ == "__main__": build()
