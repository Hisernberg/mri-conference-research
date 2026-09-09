#!/usr/bin/env python3
"""Private Kaggle preparation and development experiments, with a closed locked test."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from qmmf.config import ExperimentConfig, MODALITIES
from qmmf.dataset import CaseCache, SliceDataset, canonical_modality_order
from qmmf.evaluate import evaluate_cases, evaluate_all_subsets, validation_metrics
from qmmf.grouping import group_means, group_quality_medians
from qmmf.labels import LabelScheme
from qmmf.manifest import build_manifest, discover_cases, read_manifest
from qmmf.models import ABLATIONS, DATASET_FLAG_ABLATIONS, build_model, parameter_matched_widths
from qmmf.quality import QualityNormalizer
from qmmf.splits import Splits, build_splits, validate_splits
from qmmf.trainer import Trainer
from qmmf.utils import environment_fingerprint, seed_everything


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(native(data), sort_keys=True, indent=2, allow_nan=False))
    tmp.replace(path)


def prepare(input_dir, work):
    roots = [p.parent for p in Path(input_dir).rglob("dataset.json")
             if (p.parent / "imagesTr").is_dir() and (p.parent / "labelsTr").is_dir()]
    if len(roots) != 1:
        raise ValueError(f"Expected one labeled MSD root, found {roots}")
    root = roots[0]; dest = Path(work) / "prepared"; dest.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "dataset.json").read_text())
    order = canonical_modality_order(metadata)
    save(dest / "source_dataset.json", metadata)
    save(dest / "channel_order.json", {"source": metadata["modality"], "canonical": MODALITIES, "permutation": order})
    start = time.monotonic()
    manifest, report = build_manifest(root, progress=True, workers=2)
    manifest.to_csv(dest / "manifest.csv", index=False)
    save(dest / "dataset_fingerprint.json", report)
    if not report["gate_pass"]:
        raise RuntimeError("Dataset integrity checks failed")
    if len(manifest) != 484:
        raise RuntimeError(f"Expected 484 labeled MSD cases, found {len(manifest)}; review before changing the cohort")
    splits = build_splits(manifest, seed=42)
    splits.save(dest / "splits.json")
    scheme = LabelScheme(name=report["label_scheme"], **report["label_scheme_mapping"])
    cache = CaseCache(dest / "cache_v2", scheme, max_memory_cases=0)
    cases = discover_cases(root)
    def build(case):
        return cache.build(case["case_id"], case["image_path"], case["label_path"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, _ in enumerate(pool.map(build, cases), 1):
            if i % 25 == 0:
                print(f"Cached {i}/{len(cases)}", flush=True)
    # Deterministic lists are frozen without using model outputs.
    rng = np.random.default_rng(20260909)
    inner = list(splits.folds["0"]["inner_val"])
    outer = list(splits.folds["0"]["outer_val"])
    rng.shuffle(inner); rng.shuffle(outer)
    save(dest / "development_cohorts.json", {"validation": inner[:4],
        "outer_evaluation": outer, "robustness": outer[:8], "locked_test_opened": False})
    save(dest / "preparation_complete.json", {"completed": True, "cases": len(cases),
        "split_hash": splits.split_hash, "cache_version": 2,
        "wall_seconds": time.monotonic() - start,
        "cache_bytes": sum(p.stat().st_size for p in (dest / "cache_v2").glob("*.npz"))})
    print("PREPARATION COMPLETE", dest, flush=True)


VARIANTS = {"qmmf": ("qmmf_net", "A0"), "hemis": ("hemis25d", "A0"),
    "unet25d": ("unet25d", "A0"), "no_quality": ("qmmf_net", "A5"),
    "no_variance": ("qmmf_net", "A6"), "no_max": ("qmmf_net", "A7"),
    "no_consistency": ("qmmf_net", "A10"), "shuffled_quality": ("qmmf_net", "A17"),
    "matched_moment_fusion": ("qmmf_net", "A16")}


def find_prepared(input_dir):
    hits = [p.parent for p in Path(input_dir).rglob("preparation_complete.json")]
    if len(hits) != 1:
        raise ValueError(f"Attach the completed private CPU preparation output; found {hits}")
    return hits[0]


def context(prepared, fold=0, phase="pilot", training_slabs=None):
    report = json.loads((prepared / "dataset_fingerprint.json").read_text())
    scheme = LabelScheme(name=report["label_scheme"], **report["label_scheme_mapping"])
    cache = CaseCache(prepared / "cache_v2", scheme, max_memory_cases=2, training_slabs=training_slabs)
    frozen = Path(__file__).resolve().parents[1] / "configs/grouped_v2"
    provenance = json.loads((frozen / "provenance.json").read_text())
    if provenance["manifest_hash"] != report["manifest_hash"]:
        raise ValueError("Frozen groups were audited against a different dataset")
    splits = Splits.load(frozen / "splits.json")
    validate_splits(splits)
    if provenance["split_hash"] != splits.split_hash:
        raise ValueError("Frozen split/provenance mismatch")
    if not getattr(splits, "case_to_group", None):
        raise ValueError("Case-ID-only splits cross intensity-rescaled MRI copies. A verified group-aware split is required before training.")
    pool = json.loads((frozen / "development_cohorts.json").read_text())["folds"][str(fold)]
    cohorts = {"validation": pool[f"validation_{phase}"], "outer_evaluation": pool["outer_evaluation"],
               "robustness": pool[f"robustness_{phase}"], "fold": fold, "phase": phase,
               "locked_test_opened": False, "unit": "conservative_image_similarity_group"}
    manifest = read_manifest(prepared / "manifest.csv")
    if set(manifest.case_id) != set(splits.case_to_group):
        raise ValueError("Frozen grouping does not cover the input manifest")
    spacing = {str(r.case_id): ast.literal_eval(str(r.zooms)) for r in manifest.itertuples()}
    assert not set(splits.locked_test) & set(cohorts["outer_evaluation"])
    for key, role in [("validation", "inner_val"), ("robustness", "outer_val")]:
        ids = cohorts[key]
        assert set(ids) <= set(splits.folds[str(fold)][role])
        assert len(ids) == len({splits.case_to_group[c] for c in ids})
    return report, cache, splits, cohorts, spacing


def worker(config_path, prepared, name, phase="pilot", training_slabs=None):
    torch.set_num_threads(2)
    cfg = ExperimentConfig.load(config_path)
    report, cache, splits, cohorts, spacing = context(prepared, cfg.fold, phase, training_slabs)
    assert cfg.data.split_hash == splits.split_hash
    run = cfg.run_dir(); run.mkdir(parents=True, exist_ok=True)
    if (run / "completed.json").exists():
        print("Already completed", name, flush=True); return
    train_ids = splits.folds[str(cfg.fold)]["train"]
    norm = QualityNormalizer.fit(group_quality_medians(train_ids, cache, splits.case_to_group))
    save(run / "quality_normalizer.json", {"training_case_ids": train_ids,
        "training_group_ids": sorted({splits.case_to_group[c] for c in train_ids}),
        "fit_unit": "within-group median descriptor, robust scaler across groups", "normalizer": norm.to_dict()})
    dataset = SliceDataset(train_ids, cache, cfg.data, quality_norm=norm,
        curriculum=cfg.train.modality_curriculum, length=cfg.train.steps_per_epoch * cfg.train.batch_size,
        seed=cfg.train.seed, case_to_group=splits.case_to_group,
        **DATASET_FLAG_ABLATIONS.get(cfg.ablation, {}))
    save(run / "training_sampling.json", {"unit": "uniform group then uniform case",
        "n_groups": len(dataset.group_ids), "n_cases": len(dataset.case_ids),
        "quality_donors": dataset.quality_donors})
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires an attached supported accelerator")
    # A real CUDA operation detects incompatible driver/wheel/architecture before training.
    assert torch.isfinite(torch.ones(4, device=device).sum())
    selected_subsets = [MODALITIES, ("t1", "t2", "flair"), ("t1", "t1ce", "t2")]
    def validate(model, epoch):
        return validation_metrics(model, cohorts["validation"], cache, cfg, spacing,
                                  device=device, subsets=selected_subsets)
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(cfg, dataset, validate, device=device,
        logger=lambda text: print(f"[{name}] {text}", flush=True))
    state = trainer.fit(resume=True)
    fit_seconds = time.monotonic() - start
    if state.best_epoch < 0:
        raise RuntimeError("No valid selected checkpoint")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.model.quality_normalizer = QualityNormalizer.from_dict(checkpoint["quality_normalizer"])
    result = evaluate_cases(trainer.model, cohorts["outer_evaluation"], cache, cfg,
                            spacing, device=device, detailed_metrics=False, progress=f"{name} outer")
    outer_seconds = time.monotonic() - start - fit_seconds
    grouped = group_means(result["per_case_macro"], splits.case_to_group)
    primary_dice = float(np.mean(list(grouped.values())))
    pd.DataFrame([r.__dict__ for r in result["rows"]]).to_csv(run / "outer_case_metrics.csv", index=False)
    save(run / "outer_summary.json", {"scope": f"development_fold{cfg.fold}", "not_confirmatory": True,
        "summary": result["summary"], "per_case_macro": result["per_case_macro"],
        "per_group_macro": grouped, "primary_group_macro_dice": primary_dice,
        "primary_unit": "conservative_image_similarity_group",
        "locked_test_opened": False})
    # All 15 combinations, on the same prespecified eight development cases.
    subsets = evaluate_all_subsets(trainer.model, cohorts["robustness"], cache, cfg,
                                   spacing, device=device, progress=f"{name} robustness")
    from qmmf.shapley import shapley_from_subset_table, efficiency_gap
    phis = {cid: shapley_from_subset_table(table) for cid, table in subsets["per_case"].items()}
    from qmmf.subsets import subset_key
    for cid, phi in phis.items():
        assert abs(efficiency_gap(phi, subsets["per_case"][cid][subset_key(MODALITIES)])) < 1e-8
    save(run / "modality_subsets.json", {"scope": f"development_subset_n{len(cohorts['robustness'])}",
        "one_representative_per_group": True, **subsets, "shapley": phis})
    save(run / "completed.json", {"completed": True, "name": name,
        "scope": f"development_fold{cfg.fold}", "phase": phase, "seed": cfg.seed, "fold": cfg.fold,
        "config_hash": cfg.config_hash(), "best_epoch": state.best_epoch,
        "inner_selection_score": state.best_score, "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "outer_macro_dice": primary_dice, "outer_case_macro_dice": result["summary"]["macro_dice"],
        "outer_n_groups": len(grouped),
        "robustness_macro_dice": float(np.mean(list(subsets["mean_over_subsets"].values()))),
        "robustness_n_cases": len(cohorts["robustness"]), "outer_n_cases": len(cohorts["outer_evaluation"]),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
        "fit_and_validation_seconds": fit_seconds, "outer_evaluation_seconds": outer_seconds,
        "subset_evaluation_seconds": time.monotonic() - start - fit_seconds - outer_seconds,
        "wall_seconds": time.monotonic() - start, "locked_test_opened": False})
    print("COMPLETED", name, flush=True)


def find_training_slabs(input_dir, report, splits):
    """Require completed, fully verified lossless staging before the longer study."""
    hits = list(Path(input_dir).rglob("training_slabs_complete.json"))
    if len(hits) != 1:
        raise ValueError(f"Extended study requires one verified training-chunk input, found {len(hits)}")
    path = hits[0]; complete = json.loads(path.read_text())
    if not (complete["completed"] and complete["voxel_equality_verified_for_all_cases"]
            and complete["manifest_hash"] == report["manifest_hash"]):
        raise ValueError("Training cache is incomplete or belongs to a different dataset")
    inventory = path.parent / "case_inventory.csv"
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != complete["case_inventory_sha256"]:
        raise ValueError("Training cache inventory hash mismatch")
    frame = pd.read_csv(inventory)
    if set(frame.case_id) != set(splits.case_to_group) or len(frame) != len(splits.case_to_group):
        raise ValueError("Training cache case universe mismatch")
    if set(p.stem for p in (path.parent / "cache").glob("*.npz")) != set(frame.case_id):
        raise ValueError("A training slab archive is missing")
    return path.parent / "cache", complete


def study(input_dir, work, scope, budget_seconds=15600, seed=42, fold=0,
          epochs=None, steps=None, val_every=4, variants=None):
    torch.set_num_threads(2); seed_everything(seed)
    prepared = find_prepared(input_dir)
    report, cache, splits, cohorts, spacing = context(prepared, fold, scope)
    slab_path, slab_provenance = (find_training_slabs(input_dir, report, splits)
                                  if scope == "study" else (None, None))
    dest = Path(work) / "segmentation_results"; dest.mkdir(parents=True, exist_ok=True)
    save(dest / "environment.json", environment_fingerprint())
    save(dest / "dataset_fingerprint.json", report)
    save(dest / "development_cohorts.json", cohorts)
    splits.save(dest / "splits.json")
    frozen = Path(__file__).resolve().parents[1] / "configs/grouped_v2"
    save(dest / "grouping_provenance.json", json.loads((frozen / "provenance.json").read_text()))
    if slab_provenance is not None:
        save(dest / "training_cache_provenance.json", slab_provenance)
    cache_note = (f"slab8:{slab_provenance['case_inventory_sha256']}" if slab_provenance else "volume_npz_v2")
    base = ExperimentConfig().merged({"out_dir": str(dest / "qmmf_runs"),
        "seed": seed, "fold": fold, "notes": f"grouped_v2 {scope}; fresh weights; group-weighted sampling and primary metrics; training_cache={cache_note}",
        "data.root": str(prepared), "data.split_hash": splits.split_hash,
        "data.manifest_hash": report["manifest_hash"],
        "train.max_epochs": epochs or (24 if scope == "study" else 8),
        "train.steps_per_epoch": steps or (100 if scope == "study" else 40),
        "train.batch_size": 4, "train.accumulation_steps": 2,
        "train.val_every": val_every, "train.num_workers": 1,
        "train.early_stopping_patience": 8, "train.seed": seed})
    names = variants or (list(VARIANTS) if scope == "study" else ["qmmf", "hemis", "no_quality"])
    if len(set(names)) != len(names) or not set(names) <= set(VARIANTS):
        raise ValueError("Invalid or duplicate variant names")
    configs = {}
    for name in names:
        model, ablation = VARIANTS[name]
        cfg = ABLATIONS[ablation].apply(base.merged({"model": model}))
        if name == "matched_moment_fusion":
            matched = parameter_matched_widths(base, "equal_mean_var", base_widths=base.net.widths,
                                                channel_multiple=8, tolerance=.02)
            if not matched["within_tolerance"]:
                raise RuntimeError("Parameter-matched control is outside tolerance")
            cfg = cfg.merged({"net.widths": matched["widths"]})
            save(dest / "capacity_match.json", matched)
        p = dest / "configs" / f"{name}.json"; cfg.save(p); configs[name] = p
    save(dest / "protocol_lock.json", {"scope": scope, "names": names,
        "training_read_format": cache_note,
        "fold": fold, "seed": seed, "sampling_unit": "uniform group then uniform case",
        "primary_metric_unit": "conservative_image_similarity_group",
        "configs": {n: ExperimentConfig.load(p).to_dict() for n, p in configs.items()},
        "split_hash": splits.split_hash, "test_access": "locked_test_closed",
        "robustness_case_ids": cohorts["robustness"], "budget_seconds": budget_seconds,
        "claim_status": "single-fold single-seed development evidence; not a completed conference study"})
    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise RuntimeError("No GPU; refusing to silently run a different experiment")
    start = time.monotonic(); pending = list(names); active = {}; failures = {}
    while pending or active:
        for gpu in range(min(n_gpu, 2)):
            if gpu in active or not pending:
                continue
            if time.monotonic() - start >= budget_seconds:
                break
            name = pending.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            cmd = [sys.executable, str(Path(__file__)), "--scope", "worker", "--prepared", str(prepared),
                    "--config", str(configs[name]), "--name", name, "--phase", scope]
            if slab_path is not None:
                cmd.extend(["--training-slabs", str(slab_path)])
            active[gpu] = (name, subprocess.Popen(cmd, env=env))
            print("START", name, "GPU", gpu, flush=True)
        for gpu, (name, process) in list(active.items()):
            code = process.poll()
            if code is not None:
                if code:
                    failures[name] = code
                del active[gpu]
        if time.monotonic() - start >= budget_seconds:
            # SIGTERM is confined to our own worker processes; last.pt was saved per epoch.
            for name, process in active.values():
                process.terminate(); process.wait(timeout=30)
                failures[name] = "budget_exhausted_checkpoint_available"
            active.clear(); break
        time.sleep(5)
    completed = []
    for name, p in configs.items():
        run = ExperimentConfig.load(p).run_dir()
        if (run / "completed.json").exists():
            completed.append(json.loads((run / "completed.json").read_text()))
    pd.DataFrame(completed).to_csv(dest / "model_comparison.csv", index=False)
    save(dest / "session_status.json", {"completed_variants": [r["name"] for r in completed],
        "planned_variants": names, "failures": failures, "pending": pending,
        "all_planned_completed": len(completed) == len(names), "scope": scope,
        "locked_test_opened": False, "wall_seconds": time.monotonic() - start})
    print(pd.DataFrame(completed).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input"); p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--scope", choices=["prepare", "pilot", "study", "worker"], default="pilot")
    p.add_argument("--prepared", type=Path); p.add_argument("--config", type=Path); p.add_argument("--name")
    p.add_argument("--budget-seconds", type=int, default=15600)
    p.add_argument("--phase", choices=["pilot", "study"], default="pilot")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int); p.add_argument("--steps", type=int)
    p.add_argument("--val-every", type=int, default=4)
    p.add_argument("--variants", nargs="+")
    p.add_argument("--training-slabs", type=Path)
    a = p.parse_args()
    if a.scope == "prepare":
        prepare(a.input, a.work)
    elif a.scope == "worker":
        worker(a.config, a.prepared, a.name, a.phase, a.training_slabs)
    else:
        study(a.input, a.work, a.scope, a.budget_seconds, a.seed, a.fold,
              a.epochs, a.steps, a.val_every, a.variants)


if __name__ == "__main__":
    main()
