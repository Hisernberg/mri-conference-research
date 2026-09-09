#!/usr/bin/env python3
"""Combine independently verified grouped studies without selecting favorable seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["outer_macro_dice", "mean_subset_macro_dice", "worst_subset_macro_dice"]
VARIANTS = ["qmmf", "hemis", "unet25d", "no_quality", "no_variance", "no_max",
            "no_consistency", "shuffled_quality", "matched_moment_fusion"]


def read(path):
    return json.loads(Path(path).read_text())


def repeated_seed_summary(frames, metric, draws=10_000, seed=20260909):
    """Resample groups; retain all fixed training seeds inside every paired draw."""
    first = next(iter(frames.values())).sort_index().sort_index(axis=1)
    seeds, arrays = sorted(frames), []
    if len(first) < 2 or first.columns.duplicated().any() or first.index.duplicated().any():
        raise ValueError("Need unique aligned groups and models")
    for training_seed in seeds:
        frame = frames[training_seed].sort_index().sort_index(axis=1)
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("Group/model alignment differs across training seeds")
        if not np.isfinite(frame.values).all() or ((frame.values < 0) | (frame.values > 1)).any():
            raise ValueError("Invalid group Dice values")
        arrays.append(frame.values)
    values = np.stack(arrays)  # seed, group, model
    seed_metrics = values.mean(axis=1)
    group_seed_mean = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(first), size=(draws, len(first)))
    boot = group_seed_mean[indices].mean(axis=1)
    means = seed_metrics.mean(axis=0)
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    rows = [{"model": name, "metric": metric, "mean": float(means[i]),
             "seed_sd": float(seed_metrics[:, i].std(ddof=1)) if len(seeds) > 1 else None,
             "ci_low": float(lo[i]), "ci_high": float(hi[i]),
             "n_groups": len(first), "n_training_seeds": len(seeds)}
            for i, name in enumerate(first.columns)]
    contrasts = []
    if "qmmf" in first:
        reference = first.columns.get_loc("qmmf")
        family_size = len(first.columns) - 1
        for i, name in enumerate(first.columns):
            if name == "qmmf":
                continue
            differences = boot[:, reference] - boot[:, i]
            ci = np.quantile(differences, [.025, .975])
            adjusted = np.quantile(differences, [.025 / family_size, 1 - .025 / family_size])
            contrasts.append({"reference": "qmmf", "control": name, "metric": metric,
                "difference": float(means[reference] - means[i]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                "bonferroni_ci_low": float(adjusted[0]), "bonferroni_ci_high": float(adjusted[1]),
                "comparison_family_size": family_size, "family_definition": "all prespecified controls versus QMMF for this endpoint",
                "n_groups": len(first), "n_training_seeds": len(seeds)})
    seed_rows = [{"model": name, "seed": training_seed, "metric": metric,
                  "value": float(seed_metrics[j, i]), "n_groups": len(first)}
                 for j, training_seed in enumerate(seeds) for i, name in enumerate(first.columns)]
    return rows, contrasts, seed_rows


def combine(sources, dest, expected_seeds, expected_folds, variants, draws=10_000):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    expected = {(fold, seed) for fold in expected_folds for seed in expected_seeds}
    indexed, split_hash, common_configs, common_cache = {}, None, None, None
    for folder in map(Path, sources):
        verification, protocol = read(folder / "verification.json"), read(folder / "protocol_lock.json")
        if protocol["scope"] != "study" or not verification["all_planned_completed"]:
            raise ValueError("Only complete, independently verified extended studies may be combined")
        if set(protocol["names"]) != set(variants):
            raise ValueError("A study omits a prespecified variant")
        key = (int(protocol["fold"]), int(protocol["seed"]))
        if key in indexed or key not in expected:
            raise ValueError("Repeated or unexpected fold/seed study")
        if split_hash is not None and verification["split_hash"] != split_hash:
            raise ValueError("Studies used different frozen partitions")
        split_hash = verification["split_hash"]
        if verification["locked_test_opened"]:
            raise ValueError("Unexpected protected-test access")
        signatures = {}
        for name in variants:
            cfg = json.loads(json.dumps(protocol["configs"][name]))
            if cfg["fold"] != key[0] or cfg["seed"] != key[1] or cfg["train"]["seed"] != key[1]:
                raise ValueError("Configuration seed/fold differs from its protocol")
            for field in ["seed", "fold", "out_dir", "notes"]:
                cfg.pop(field)
            cfg["train"].pop("seed")
            cfg["data"].pop("root")
            signatures[name] = json.dumps(cfg, sort_keys=True)
        if common_configs is not None and signatures != common_configs:
            raise ValueError("Training/evaluation configurations changed between repeated studies")
        common_configs = signatures
        if common_cache is not None and protocol["training_read_format"] != common_cache:
            raise ValueError("Training cache provenance changed between repeated studies")
        common_cache = protocol["training_read_format"]
        indexed[key] = folder
    if set(indexed) != expected:
        raise ValueError(f"Missing required fold/seed studies: {sorted(expected - set(indexed))}")
    rows, contrasts, seed_rows, run_rows = [], [], [], []
    for metric in METRICS:
        frames = {}
        for training_seed in expected_seeds:
            pieces = [pd.read_csv(indexed[(fold, training_seed)] / f"{metric}_by_group.csv", index_col="group_id")
                      for fold in expected_folds]
            frame = pd.concat(pieces)
            if frame.index.duplicated().any():
                raise ValueError("A group appears in multiple outer folds")
            if set(frame.columns) != set(variants):
                raise ValueError("Metric table omits a prespecified variant")
            frames[training_seed] = frame[variants]
            frame.to_csv(dest / f"{metric}_seed{training_seed}_by_group.csv", index_label="group_id")
        a, b, c = repeated_seed_summary(frames, metric, draws)
        rows.extend(a); contrasts.extend(b); seed_rows.extend(c)
    for (fold, seed), folder in sorted(indexed.items()):
        table = pd.read_csv(folder / "model_comparison.csv")
        if set(table.name) != set(variants) or len(table) != len(variants):
            raise ValueError("Run completion table is incomplete")
        run_rows.extend(table.assign(fold=fold, seed=seed).to_dict("records"))
    pd.DataFrame(rows).to_csv(dest / "repeated_seed_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(dest / "paired_repeated_seed_deltas.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(dest / "seed_specific_metrics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(dest / "all_model_runs.csv", index=False)
    record = {"all_required_completed": True, "seeds": expected_seeds, "folds": expected_folds,
        "variants": variants, "completed_model_fits": len(run_rows), "split_hash": split_hash,
        "bootstrap_draws": draws, "bootstrap_seed": 20260909,
        "bootstrap_unit": "conservative image group; all fixed training seeds retained in every draw",
        "inference": "descriptive conditional on the observed training seeds; no verified patient or external generalization claim",
        "multiplicity": "Bonferroni percentile intervals across prespecified controls separately for each endpoint",
        "seed_sd": "sample SD across seed-specific pooled group metrics; seeds are not additional cases",
        "locked_test_opened": False, "sources": [str(p) for p in map(Path, sources)]}
    (dest / "combination_verification.json").write_text(json.dumps(record, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="+", type=Path); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--draws", type=int, default=10_000)
    a = p.parse_args(); combine(a.sources, a.output, a.seeds, a.folds, a.variants, a.draws)
