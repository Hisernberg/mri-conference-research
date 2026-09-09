#!/usr/bin/env python3
"""Bounded Kaggle throughput measurement; all benchmark weights are discarded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from conference_run import context, find_prepared, find_training_slabs, save
from qmmf.config import ExperimentConfig
from qmmf.dataset import SliceDataset
from qmmf.grouping import group_quality_medians
from qmmf.quality import QualityNormalizer
from qmmf.trainer import Trainer
from qmmf.utils import environment_fingerprint


def worker(input_dir, out, model):
    torch.set_num_threads(2)
    prepared = find_prepared(input_dir)
    report, original, splits, _, _ = context(prepared)
    slab_path, slab_provenance = find_training_slabs(input_dir, report, splits)
    _, fast, _, _, _ = context(prepared, training_slabs=slab_path)
    ids = splits.folds["0"]["train"]
    norm = QualityNormalizer.fit(group_quality_medians(ids, original, splits.case_to_group))
    cfg = ExperimentConfig().merged({"model": model, "train.max_epochs": 120,
        "train.steps_per_epoch": 100, "train.batch_size": 4, "train.accumulation_steps": 2,
        "train.num_workers": 1, "out_dir": str(out / model),
        "data.split_hash": splits.split_hash, "data.manifest_hash": report["manifest_hash"]})
    order = ["volume", "slab8"] if model == "qmmf_net" else ["slab8", "volume"]
    results = []
    for kind in order:
        cache = original if kind == "volume" else fast
        ds = SliceDataset(ids, cache, cfg.data, quality_norm=norm,
            length=400, seed=42, case_to_group=splits.case_to_group)
        trainer = Trainer(cfg, ds, lambda model, epoch: {}, device=torch.device("cuda"),
                          run_dir=out / model / kind)
        # Warm CUDA and the real model with 10 microbatches; then time 100.
        ds.length = 40
        trainer.train_epoch(25)
        ds.length = 400
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        metrics = trainer.train_epoch(25)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if not np.isfinite(metrics["total"]):
            raise FloatingPointError("Benchmark produced non-finite training loss")
        results.append({"model": model, "training_cache": kind, "timed_microbatches": 100,
            "warmup_microbatches": 10, "batch_size": 4, "accumulation_steps": 2,
            "seconds": elapsed, "seconds_per_microbatch": elapsed / 100,
            "central_slices_per_second": 400 / elapsed,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "consistency_active": True, "training_loss_finite": True})
        del trainer, ds
        torch.cuda.empty_cache()
    save(out / f"{model}.json", {"results": results, "order": order,
        "split_hash": splits.split_hash, "training_groups": len({splits.case_to_group[c] for c in ids}),
        "training_case_ids": ids, "training_cache_inventory": slab_provenance["case_inventory_sha256"],
        "weights_discarded": True, "validation_or_test_evaluated": False})
    print(json.dumps(results, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/io_benchmark"))
    p.add_argument("--worker", choices=["qmmf_net", "hemis25d"])
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(a.input, a.out, a.worker); return
    save(a.out / "environment.json", environment_fingerprint())
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires the documented two-T4 environment")
    processes = []
    for gpu, model in enumerate(["qmmf_net", "hemis25d"]):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__)),
            "--input", a.input, "--out", str(a.out), "--worker", model], env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Benchmark worker failed: {codes}")
    save(a.out / "completion.json", {"completed": True, "models": ["qmmf_net", "hemis25d"],
        "scope": "training throughput only; no reusable model fit or validation/test result",
        "timed_steps_per_model_and_format": 100, "warmup_steps_per_model_and_format": 10})


if __name__ == "__main__":
    main()
