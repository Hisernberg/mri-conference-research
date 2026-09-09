"""Single-GPU worker and dual-T4 launcher (plan 11.1-11.2).

Two T4s are two 16 GB devices, not one 32 GB device. The default strategy is
experiment-level parallelism: GPU 0 trains one (model, fold, seed) while GPU 1
trains another. Each worker claims from the shared queue, writes into its own
run directory and appends one ledger row.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import ExperimentConfig
from .ledger import ExperimentEntry, ExperimentQueue, Ledger
from .utils import code_hash, environment_fingerprint, seed_everything


def build_experiment_queue(
    queue_path: str | Path,
    models: List[str],
    folds: List[int],
    seeds: List[int],
    ablations: Optional[List[str]] = None,
    label_fractions: Optional[List[float]] = None,
    base_cfg: Optional[ExperimentConfig] = None,
    stage: str = "development",
) -> int:
    """Populate the queue with the cross-product the run matrix calls for."""
    base = base_cfg or ExperimentConfig()
    ablations = ablations or ["A0"]
    label_fractions = label_fractions or [1.0]
    entries: List[ExperimentEntry] = []
    for model in models:
        for ablation in ablations:
            for fold in folds:
                for seed in seeds:
                    for lf in label_fractions:
                        cfg = base.merged({
                            "model": model, "ablation": ablation, "fold": fold,
                            "seed": seed, "label_fraction": lf, "stage": stage,
                            "train.seed": seed,
                        })
                        entries.append(ExperimentEntry(
                            run_id=cfg.run_id(), model=model, ablation=ablation,
                            fold=fold, seed=seed, label_fraction=lf, stage=stage,
                        ))
    return ExperimentQueue(queue_path).add(entries)


# --------------------------------------------------------------------------- #
def run_worker(
    gpu: int,
    queue_path: str | Path,
    ledger_path: str | Path,
    run_one: Callable[[ExperimentConfig, int], Dict[str, Any]],
    base_cfg: Optional[ExperimentConfig] = None,
    max_runs: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    src_root: Optional[str | Path] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Claim and execute experiments until the queue empties or a budget expires.

    `run_one(cfg, gpu) -> metrics dict` does the actual training/evaluation.
    A run that raises is recorded as failed and the worker continues, so one
    OOM does not take down the whole session.
    """
    log = logger or (lambda m: print(f"[gpu{gpu}] {m}", flush=True))
    queue = ExperimentQueue(queue_path)
    ledger = Ledger(ledger_path)
    base = base_cfg or ExperimentConfig()
    ch = code_hash(src_root) if src_root else ""
    env = environment_fingerprint()
    started = time.time()
    done: List[str] = []

    while True:
        if max_runs is not None and len(done) >= max_runs:
            log(f"reached max_runs={max_runs}")
            break
        if time_budget_seconds is not None and time.time() - started > time_budget_seconds:
            log("time budget exhausted; stopping cleanly so the notebook can commit")
            break

        entry = queue.claim(gpu)
        if entry is None:
            log("queue empty")
            break
        if entry["run_id"] in ledger.completed_run_ids():
            log(f"{entry['run_id']} already completed; skipping")
            queue.release(entry["run_id"], "completed")
            continue

        cfg = base.merged({
            "model": entry["model"], "ablation": entry["ablation"],
            "fold": entry["fold"], "seed": entry["seed"],
            "label_fraction": entry["label_fraction"], "stage": entry["stage"],
            "train.seed": entry["seed"],
        })
        row = {
            "run_id": entry["run_id"], "gpu": gpu, "model": cfg.model,
            "ablation": cfg.ablation, "fold": cfg.fold, "seed": cfg.seed,
            "label_fraction": cfg.label_fraction, "stage": cfg.stage,
            "config_hash": cfg.config_hash(),
            "manifest_hash": cfg.data.manifest_hash,
            "split_hash": cfg.data.split_hash,
            "code_hash": ch, "git_commit": env.get("git_commit", ""),
        }
        log(f"start {entry['run_id']}")
        t0 = time.time()
        try:
            seed_everything(cfg.train.seed)
            metrics = run_one(cfg, gpu)
            row.update(metrics)
            row["status"] = "completed"
            row["wall_seconds"] = time.time() - t0
            ledger.append(row)
            queue.release(entry["run_id"], "completed")
            done.append(entry["run_id"])
            log(f"done {entry['run_id']} in {row['wall_seconds']:.0f}s")
        except Exception as exc:  # noqa: BLE001 - failures are evidence, not noise
            row["wall_seconds"] = time.time() - t0
            status = "resource_rejected" if _is_oom(exc) else "failed"
            row["status"] = status
            ledger.record_failure(row, exc)
            queue.release(entry["run_id"], status)
            log(f"FAILED {entry['run_id']}: {type(exc).__name__}: {exc}")
            _empty_cache()
    return done


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "cuda oom" in text


def _empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
def launch_dual_gpu(
    worker_script: str | Path,
    queue_path: str | Path,
    ledger_path: str | Path,
    extra_args: Optional[List[str]] = None,
    gpus: tuple = (0, 1),
) -> List[subprocess.Popen]:
    """Start one independent process per T4.

    Independent processes, separate 16 GB VRAM spaces, one shared ledger. This
    is not DDP: DDP would duplicate model state on both GPUs for a single run,
    which is the wrong trade for a study that needs many short runs.
    """
    procs: List[subprocess.Popen] = []
    for gpu in gpus:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable, str(worker_script),
            "--gpu", "0",                     # remapped to 0 inside the process
            "--queue", str(queue_path),
            "--ledger", str(ledger_path),
        ] + (extra_args or [])
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)                          # stagger the dataset cache warmup
    return procs


def wait_for(procs: List[subprocess.Popen], poll: float = 30.0) -> List[int]:
    codes = []
    while any(p.poll() is None for p in procs):
        time.sleep(poll)
    for p in procs:
        codes.append(p.returncode)
    return codes
