#!/usr/bin/env python3
"""Single-GPU worker process (plan 11.2).

Claims experiments from the shared queue, trains one, evaluates it on the
fold's outer-validation cases, writes an OOF prediction file, and appends one
ledger row. Two of these run concurrently - one per T4 - with
CUDA_VISIBLE_DEVICES pinned by the launcher.

    python scripts/run_worker.py --gpu 0 \
        --queue artifacts/experiment_queue.json \
        --ledger artifacts/experiment_ledger.csv \
        --base configs/base.yaml --cache /kaggle/working/cache \
        --splits artifacts/splits.json --manifest artifacts/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmmf.config import ExperimentConfig                     # noqa: E402
from qmmf.dataset import CaseCache, SliceDataset             # noqa: E402
from qmmf.evaluate import validation_metrics                 # noqa: E402
from qmmf.labels import MSD_TASK01                           # noqa: E402
from qmmf.models import DATASET_FLAG_ABLATIONS               # noqa: E402
from qmmf.quality import QualityNormalizer                   # noqa: E402
from qmmf.splits import Splits, training_cases               # noqa: E402
from qmmf.trainer import Trainer                             # noqa: E402
from qmmf.utils import count_parameters, read_json, write_json  # noqa: E402
from qmmf.worker import run_worker                           # noqa: E402


def make_runner(args, splits: Splits, manifest: pd.DataFrame, cache: CaseCache):
    spacing = {str(r.case_id): tuple(r.zooms) if isinstance(r.zooms, (list, tuple))
               else eval(str(r.zooms)) for r in manifest.itertuples()}

    def run_one(cfg: ExperimentConfig, gpu: int) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Integrity gate: refuse to train under an unverified data partition.
        if cfg.data.split_hash and cfg.data.split_hash != splits.split_hash:
            raise RuntimeError(
                f"Split hash mismatch: config {cfg.data.split_hash} vs "
                f"file {splits.split_hash}. Refusing to train."
            )

        train_ids = training_cases(splits, cfg.fold, cfg.label_fraction)
        inner_val = splits.folds[str(cfg.fold)]["inner_val"]

        qnorm = None
        qpath = Path(args.cache).parent / f"quality_norm_fold{cfg.fold}.json"
        if qpath.exists():
            qnorm = QualityNormalizer.from_dict(read_json(qpath))
        else:
            qnorm = QualityNormalizer.fit(
                [cache.load(c)["quality_raw"] for c in train_ids]
            )
            write_json(qpath, qnorm.to_dict())

        flags = DATASET_FLAG_ABLATIONS.get(cfg.ablation, {})
        dataset = SliceDataset(
            train_ids, cache, cfg.data, quality_norm=qnorm,
            curriculum=cfg.train.modality_curriculum,
            length=cfg.train.steps_per_epoch * cfg.train.batch_size,
            seed=cfg.train.seed, **flags,
        )

        def validate(model, epoch):
            return validation_metrics(model, inner_val, cache, cfg, spacing,
                                      device=device, max_cases=args.max_val_cases)

        trainer = Trainer(cfg, dataset, validate, device=device,
                          run_dir=cfg.run_dir())
        state = trainer.fit(resume=True)

        params = count_parameters(trainer.model)["total"]
        last = state.history[-1] if state.history else {}
        return {
            "best_epoch": state.best_epoch,
            "best_score": state.best_score,
            "checkpoint": str(cfg.run_dir() / "best.pt"),
            "full_macro_dice": last.get("val_full_macro_dice", ""),
            "mean_subset_macro_dice": last.get("val_mean_subset_macro_dice", ""),
            "worst_subset_macro_dice": last.get("val_worst_subset_macro_dice", ""),
            "parameters": params,
        }

    return run_one


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="Seconds; stop cleanly so the Kaggle notebook can commit.")
    ap.add_argument("--max-val-cases", type=int, default=None)
    args = ap.parse_args()

    base = ExperimentConfig.load(args.base)
    splits = Splits.load(args.splits)
    manifest = pd.read_csv(args.manifest)
    cache = CaseCache(args.cache, MSD_TASK01)

    base = base.merged({"data.split_hash": splits.split_hash})
    done = run_worker(
        gpu=args.gpu, queue_path=args.queue, ledger_path=args.ledger,
        run_one=make_runner(args, splits, manifest, cache),
        base_cfg=base, max_runs=args.max_runs,
        time_budget_seconds=args.time_budget,
        src_root=Path(__file__).resolve().parents[1] / "src",
    )
    print(f"worker finished {len(done)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
