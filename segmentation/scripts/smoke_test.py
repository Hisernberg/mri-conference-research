#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data - the whole protocol in miniature.

Runs, on ~16 tiny synthetic volumes and a two-epoch budget:

  Notebook 01 path : audit -> label verification -> splits -> cache
  Notebook 02 path : experiment queue -> worker -> ledger -> checkpoint
  Notebook 03 path : locked-test inference -> 15 subsets -> exact Shapley ->
                     calibration -> paired statistics -> result card

It proves the wiring, not the science: sixteen synthetic cases and two epochs
cannot say anything about segmentation quality, and the script says so in its
own output. Runs on CPU in about a minute.

    python scripts/smoke_test.py --out /tmp/qmmf_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmmf.calibration import calibration_report, TemperatureScaler   # noqa: E402
from qmmf.config import ExperimentConfig                             # noqa: E402
from qmmf.dataset import CaseCache, SliceDataset                     # noqa: E402
from qmmf.evaluate import evaluate_all_subsets, evaluate_cases, validation_metrics  # noqa: E402
from qmmf.inference import predict_volume                            # noqa: E402
from qmmf.labels import MSD_TASK01                                   # noqa: E402
from qmmf.ledger import ExperimentQueue, Ledger                      # noqa: E402
from qmmf.manifest import build_manifest, discover_cases             # noqa: E402
from qmmf.metrics import summarize                                   # noqa: E402
from qmmf.models import build_model                                  # noqa: E402
from qmmf.quality import QualityNormalizer                           # noqa: E402
from qmmf.shapley import efficiency_gap, patient_shapley_table       # noqa: E402
from qmmf.splits import Splits, build_splits, training_cases         # noqa: E402
from qmmf.stats import apply_holm, compare_paired, non_inferiority   # noqa: E402
from qmmf.subsets import subset_key                                  # noqa: E402
from qmmf.trainer import Trainer                                     # noqa: E402
from qmmf.utils import count_parameters, seed_everything, write_json  # noqa: E402
from qmmf.worker import build_experiment_queue, run_worker           # noqa: E402


def make_synthetic_cohort(root: Path, n: int = 16, shape=(28, 28, 18)) -> Path:
    import nibabel as nib

    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.normal(100, 20, shape + (4,)).astype(np.float32)
        img[:3], img[-3:], img[:, :3], img[:, -3:] = 0, 0, 0, 0
        seg = np.zeros(shape, np.uint8)
        cz, r = 5 + i % 4, 4 + i % 3
        cx = cy = shape[0] // 2
        seg[cx - r:cx + r, cy - r:cy + r, cz:cz + 5] = 1
        seg[cx - 2:cx + 2, cy - 2:cy + 2, cz + 1:cz + 4] = 2
        if i % 4 != 0:
            seg[cx - 1:cx + 1, cy - 1:cy + 1, cz + 2:cz + 3] = 3
        # Make the tumour visible in the images so the task is not pure noise.
        img[seg > 0] += 60.0
        aff = np.diag([1.0, 1.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(img, aff), root / "imagesTr" / f"SYN_{i:03d}.nii.gz")
        nib.save(nib.Nifti1Image(seg, aff), root / "labelsTr" / f"SYN_{i:03d}.nii.gz")
    (root / "dataset.json").write_text(json.dumps({
        "name": "Synthetic", "modality": {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"},
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor",
                   "3": "enhancing tumour"},
    }))
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/qmmf_smoke")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    t0 = time.perf_counter()

    print("=" * 74)
    print("NOTEBOOK 01 PATH: audit, label verification, split lock")
    print("=" * 74)
    root = make_synthetic_cohort(out / "data")
    manifest, report = build_manifest(root, progress=False)
    manifest.to_csv(artifacts / "manifest.csv", index=False)
    print(f"  cases                {report['n_cases']}")
    print(f"  label scheme         {report['label_scheme']} "
          f"{report['label_scheme_mapping']}")
    print(f"  failed cases         {report['n_failed_cases']}")
    print(f"  exact duplicates     {len(report['duplicates']['exact_duplicate_groups'])}")
    print(f"  manifest hash        {report['manifest_hash']}")
    print(f"  G0 gate_pass         {report['gate_pass']}")
    assert report["gate_pass"], "G0 failed on synthetic data"

    splits = build_splits(manifest, seed=42)
    split_hash = splits.save(artifacts / "adult_splits.json")
    print(f"  locked test          {len(splits.locked_test)} cases")
    print(f"  development          {len(splits.development)} cases")
    print(f"  split hash           {split_hash}")
    assert Splits.load(artifacts / "adult_splits.json").split_hash == split_hash

    cache = CaseCache(out / "cache", MSD_TASK01)
    for c in discover_cases(root):
        cache.build(c["case_id"], c["image_path"], c["label_path"])
    print(f"  cache built          {len(list((out/'cache').glob('*.npz')))} files")

    cfg = ExperimentConfig().merged({
        "data.crop_size": [32, 32], "data.manifest_hash": report["manifest_hash"],
        "data.split_hash": split_hash, "data.root": str(root),
        "net.widths": [12, 24, 48], "train.max_epochs": args.epochs,
        "train.batch_size": 2, "train.accumulation_steps": 2,
        "train.num_workers": 0, "train.amp": False, "train.steps_per_epoch": 8,
        "train.early_stopping_patience": 99, "out_dir": str(out / "runs"),
        "evaluation.bootstrap_replicates": 500,
    })
    spacing = {str(r.case_id): tuple(r.zooms) for r in manifest.itertuples()}

    print()
    print("=" * 74)
    print("NOTEBOOK 02 PATH: queue, worker, ledger, checkpoint")
    print("=" * 74)
    queue_path, ledger_path = artifacts / "queue.json", artifacts / "ledger.csv"
    n_queued = build_experiment_queue(
        queue_path, models=["qmmf_net", "hemis25d"], folds=[0], seeds=[42],
        base_cfg=cfg,
    )
    print(f"  queued               {n_queued} runs")

    qnorm = QualityNormalizer.fit(
        [cache.load(c)["quality_raw"] for c in splits.folds["0"]["train"]]
    )

    def run_one(run_cfg: ExperimentConfig, gpu: int):
        train_ids = training_cases(splits, run_cfg.fold, run_cfg.label_fraction)
        inner = splits.folds[str(run_cfg.fold)]["inner_val"]
        ds = SliceDataset(train_ids, cache, run_cfg.data, quality_norm=qnorm,
                          length=run_cfg.train.steps_per_epoch * run_cfg.train.batch_size,
                          seed=run_cfg.train.seed)

        def validate(model, epoch):
            return validation_metrics(
                model, inner, cache, run_cfg, spacing, device=device,
                subsets=[("t1", "t1ce", "t2", "flair"), ("t1", "t2", "flair")],
            )

        trainer = Trainer(run_cfg, ds, validate, device=device,
                          run_dir=run_cfg.run_dir(), logger=lambda m: None)
        state = trainer.fit(resume=True)
        last = state.history[-1]
        return {
            "best_epoch": state.best_epoch, "best_score": round(state.best_score, 5),
            "checkpoint": str(run_cfg.run_dir() / "best.pt"),
            "full_macro_dice": round(last["val_full_macro_dice"], 5),
            "mean_subset_macro_dice": round(last["val_mean_subset_macro_dice"], 5),
            "worst_subset_macro_dice": round(last["val_worst_subset_macro_dice"], 5),
            "parameters": count_parameters(trainer.model)["total"],
        }

    done = run_worker(gpu=0, queue_path=queue_path, ledger_path=ledger_path,
                      run_one=run_one, base_cfg=cfg,
                      src_root=Path(__file__).resolve().parents[1] / "src",
                      logger=lambda m: print(f"  worker: {m}"))
    ledger = Ledger(ledger_path).read()
    print(f"  completed            {len(done)} runs")
    print(ledger[["model", "status", "best_epoch", "full_macro_dice",
                  "parameters", "wall_seconds"]].to_string(index=False))
    assert (ledger.status == "completed").all(), "a smoke run failed"

    # Resume must restore, not restart.
    resumed = ExperimentQueue(queue_path).summary()
    print(f"  queue after run      {resumed}")

    lock = {
        "config": cfg.to_dict(), "config_hash": cfg.config_hash(),
        "split_hash": split_hash, "manifest_hash": report["manifest_hash"],
        "decision_threshold": 0.5, "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime()),
    }
    write_json(artifacts / "model_lock.json", lock)
    print(f"  model lock written   {lock['config_hash']}")

    print()
    print("=" * 74)
    print("NOTEBOOK 03 PATH: locked test, 15 subsets, Shapley, statistics")
    print("=" * 74)
    models = {}
    for name in ("qmmf_net", "hemis25d"):
        row = ledger[(ledger.model == name) & (ledger.status == "completed")].iloc[0]
        ck = torch.load(row.checkpoint, map_location=device, weights_only=False)
        m = build_model(cfg.merged({"model": name}))
        m.load_state_dict(ck["model"])
        models[name] = m.to(device).eval()

    locked = splits.locked_test
    primary = {}
    for name, m in models.items():
        res = evaluate_cases(m, locked, cache, cfg, spacing, device=device,
                             collect_probs=True)
        primary[name] = res
        s = res["summary"]
        print(f"  {name:10s} macro={s['macro_dice']:.4f} "
              f"wt={s['dice_wt']:.4f} tc={s['dice_tc']:.4f} et={s['dice_et']:.4f} "
              f"empty-ET-refs={s['n_empty_reference_et']}")

    sub = evaluate_all_subsets(models["qmmf_net"], locked[:2], cache, cfg, spacing,
                               device=device)
    print(f"  15-subset table      {len(next(iter(sub['per_case'].values())))} subsets "
          f"x {len(sub['per_case'])} patients")

    shap = patient_shapley_table(sub["per_case"])
    gap = float(shap.efficiency_gap.abs().max())
    print(f"  Shapley efficiency   max|gap| = {gap:.2e} (must be ~0)")
    assert gap < 1e-9, "Shapley efficiency axiom violated"
    print("  mean Shapley phi     " +
          ", ".join(f"{m}={shap[m].mean():+.4f}" for m in cfg.data.modalities))

    # Calibration on the (here, tiny) predictions.
    probs = np.concatenate([primary["qmmf_net"]["probs"][c].astype(np.float32).ravel()
                            for c in locked])
    refs = np.concatenate([
        predict_volume(models["qmmf_net"], c, cache, cfg, device=device)["reference"].ravel()
        for c in locked
    ]).astype(np.float32)
    cal = calibration_report(probs, refs, cfg.evaluation.calibration_bins)
    print("  calibration          " +
          ", ".join(f"{k}={v:.4f}" for k, v in cal.items()))

    shared = sorted(set(primary["qmmf_net"]["per_case_macro"]) &
                    set(primary["hemis25d"]["per_case_macro"]))
    a = [primary["qmmf_net"]["per_case_macro"][c] for c in shared]
    b = [primary["hemis25d"]["per_case_macro"][c] for c in shared]
    cmp_ = apply_holm([compare_paired(a, b, name="qmmf_vs_hemis",
                                      n_boot=cfg.evaluation.bootstrap_replicates)])[0]
    print(f"  paired comparison    diff={cmp_.mean_difference:+.4f} "
          f"[{cmp_.ci_low:+.4f}, {cmp_.ci_high:+.4f}] "
          f"p={cmp_.p_value:.4f} p_holm={cmp_.p_holm:.4f} n={cmp_.n}")
    ni = non_inferiority(a, b, margin=cfg.evaluation.non_inferiority_margin_pp / 100,
                         n_boot=cfg.evaluation.bootstrap_replicates)
    print(f"  non-inferiority      non_inferior={ni['non_inferior']} "
          f"superior={ni['superior']} margin={ni['margin']}")

    results = {
        "protocol_version": cfg.protocol_version,
        "config_hash": cfg.config_hash(), "split_hash": split_hash,
        "manifest_hash": report["manifest_hash"],
        "locked_test_summary": {k: v["summary"] for k, v in primary.items()},
        "shapley_mean": {m: float(shap[m].mean()) for m in cfg.data.modalities},
        "shapley_max_efficiency_gap": gap,
        "calibration": cal,
        "paired_comparison": cmp_.to_dict(),
        "non_inferiority": ni,
        "WARNING": (
            "Synthetic data, 16 cases, two epochs. These numbers demonstrate that "
            "the pipeline runs end to end. They say nothing whatsoever about "
            "segmentation quality and must never appear in a manuscript."
        ),
    }
    write_json(artifacts / "smoke_results.json", results)

    print()
    print("=" * 74)
    print(f"SMOKE TEST PASSED in {time.perf_counter() - t0:.1f}s")
    print(f"artifacts: {artifacts}")
    print("=" * 74)
    print(results["WARNING"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
