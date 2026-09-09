#!/usr/bin/env python3
"""Postprocess completed Kaggle OOF predictions; no models are retrained here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/classification"


def main():
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    runs = pd.read_csv(OUT / "per_run_metrics.csv")
    efficiency = runs.groupby("model").agg(parameters=("parameters", "first"),
        parameter_counts_observed=("parameters", "nunique"), runs=("fold", "size"),
        median_fit_seconds=("wall_seconds", "median"), total_fit_seconds=("wall_seconds", "sum"),
        median_epochs_run=("epochs_run", "median"), maximum_peak_gpu_gb=("peak_gpu_gb", "max"))
    if not efficiency.parameter_counts_observed.eq(1).all():
        raise ValueError("A named model changed parameter count between runs")
    efficiency.drop(columns="parameter_counts_observed").to_csv(OUT / "compute_summary.csv")
    names = ["full", "no_attention", "no_multiscale", "no_attention_no_multiscale"]
    ids = sorted(oof.image_id.unique())
    meta = oof.drop_duplicates("image_id").set_index("image_id").loc[ids]
    arrays = {n: oof[oof.model == n].pivot(index="image_id", columns="seed", values="probability_yes").loc[ids].values >= .5 for n in names}
    groups = sorted(meta.group_id.unique())
    members = [np.where(meta.group_id.values == g)[0] for g in groups]
    y = meta.y.values

    def f1(ix, name):
        pred, truth = arrays[name][ix], y[ix, None].astype(bool)
        tp = (pred & truth).sum(0); tn = (~pred & ~truth).sum(0)
        fp = (pred & ~truth).sum(0); fn = (~pred & truth).sum(0)
        return np.mean((2*tp/np.maximum(2*tp+fp+fn,1) + 2*tn/np.maximum(2*tn+fp+fn,1))/2)

    def contrasts(ix):
        full, a0, m0, both0 = [f1(ix,n) for n in names]
        return {"attention_with_multiscale": full-a0,
                "attention_without_multiscale": m0-both0,
                "multiscale_with_attention": full-m0,
                "multiscale_without_attention": a0-both0,
                "factorial_interaction": full-a0-m0+both0}
    observed = contrasts(np.arange(len(ids)))
    rng = np.random.default_rng(20260909); draws = []
    for _ in range(2000):
        ix = np.concatenate([members[g] for g in rng.integers(len(groups),size=len(groups))])
        draws.append(contrasts(ix))
    draws = pd.DataFrame(draws)
    rows = [{"contrast": name, "macro_f1_delta": float(value),
             "ci_low": float(draws[name].quantile(.025)), "ci_high": float(draws[name].quantile(.975)),
             "analysis_scope": "exploratory factorial follow-up of Kaggle OOF predictions"}
            for name,value in observed.items()]
    pd.DataFrame(rows).to_csv(OUT / "factorial_effects.csv",index=False)
    full = oof[oof.model == "full"].copy()
    full["correct"] = (full.probability_yes >= .5) == full.y
    failures = full.groupby(["image_id","group_id","y"]).agg(
        correct_seeds=("correct","sum"),mean_probability_yes=("probability_yes","mean"),
        probability_sd=("probability_yes","std")).reset_index()
    failures.to_csv(OUT / "full_model_error_consistency.csv",index=False)
    counts = {str(k):int(v) for k,v in failures.correct_seeds.value_counts().sort_index().items()}
    (OUT / "error_analysis.json").write_text(json.dumps({"images_correct_in_n_of_3_seeds":counts,
        "unit":"unique images, not verified independent patients",
        "note":"Mean probabilities are diagnostic descriptors; headline scores use individual seeds."},indent=2))
    fig, axes = plt.subplots(1,3,figsize=(10,3))
    for ax,(seed,frame) in zip(axes,full.groupby("seed")):
        cm = confusion_matrix(frame.y,frame.probability_yes>=.5,labels=[0,1])
        ax.imshow(cm,cmap="Blues",vmin=0,vmax=141)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>80 else "black")
        ax.set_xticks([0,1],["no","yes"]); ax.set_yticks([0,1],["no","yes"])
        ax.set_title(f"Full model, seed {seed}"); ax.set_xlabel("Predicted image label")
    axes[0].set_ylabel("Reference image label")
    fig.tight_layout(); fig.savefig(OUT/"full_model_confusion_by_seed.png",dpi=250)
    fig.savefig(OUT/"full_model_confusion_by_seed.pdf");plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False));print(counts)


if __name__ == "__main__":
    main()
