"""Training loop: AMP, gradient accumulation, EMA teacher, early stopping,
composite checkpoint selection, resume (plan 8.4-8.5, 11.3).

One process trains on one T4. Experiment-level parallelism across the two GPUs
is handled by worker.py; there is no DDP on the default path, because two 16 GB
T4s are two devices, not one 32 GB device.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import SliceDataset, collate
from .ema import ModelEMA
from .losses import QMMFLoss
from .models import build_model
from .utils import seed_everything, worker_init_fn, write_json


# --------------------------------------------------------------------------- #
@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_score: float = -float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


def composite_selection_score(
    full_macro_dice: float,
    mean_subset_macro_dice: float,
    worst_subset_macro_dice: float,
) -> float:
    """Checkpoint selection rule, fixed before the main runs (plan 8.5):

        0.6 * full-modality macro Dice
      + 0.3 * mean selected-subset macro Dice
      + 0.1 * worst selected-subset macro Dice

    Calibration and pediatric performance are deliberately *not* inputs.
    """
    parts = [full_macro_dice, mean_subset_macro_dice, worst_subset_macro_dice]
    if any(not np.isfinite(p) for p in parts):
        return -float("inf")
    return 0.6 * full_macro_dice + 0.3 * mean_subset_macro_dice + 0.1 * worst_subset_macro_dice


# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        train_dataset: SliceDataset,
        validate_fn: Callable[[nn.Module, int], Dict[str, float]],
        device: Optional[torch.device] = None,
        region_weights: Optional[Sequence[float]] = None,
        run_dir: Optional[Path] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.log = logger or (lambda msg: print(msg, flush=True))
        self.run_dir = Path(run_dir or cfg.run_dir())
        self.run_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(cfg.train.seed)
        self.model = build_model(cfg).to(self.device)
        self.model.quality_normalizer = train_dataset.quality_norm
        self.criterion = QMMFLoss(cfg.loss, region_weights).to(self.device)
        self.validate_fn = validate_fn

        self.train_dataset = train_dataset
        self.loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,               # the dataset itself samples randomly
            num_workers=cfg.train.num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=False,   # epoch/curriculum must reach each worker
            drop_last=False,             # plan 8.4: drop_last=False
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
        )

        decay = cfg.train.weight_decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and self.device.type == "cuda")

        steps_per_epoch = max(math.ceil(len(self.loader) / cfg.train.accumulation_steps), 1)
        self.total_steps = steps_per_epoch * cfg.train.max_epochs
        self.warmup_steps = max(int(self.total_steps * cfg.train.warmup_frac), 1)

        self.ema: Optional[ModelEMA] = None
        if cfg.loss.consistency > 0:
            self.ema = ModelEMA(
                self.model, cfg.train.ema_decay_start, cfg.train.ema_decay_end,
                warmup_steps=self.warmup_steps, device=self.device,
            )
        self.state = TrainState()

    # ------------------------------------------------------------------ #
    def lr_at(self, step: int) -> float:
        """Linear warmup then cosine decay."""
        base = self.cfg.train.lr
        if step < self.warmup_steps:
            return base * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.train_dataset.set_epoch(epoch, self.cfg.train.max_epochs)
        cfg = self.cfg
        accum = cfg.train.accumulation_steps
        consistency_active = (
            self.ema is not None
            and epoch >= cfg.loss.consistency_warmup_frac * cfg.train.max_epochs
        )

        totals: Dict[str, float] = {}
        n_batches = 0
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.loader):
            image = batch["image"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            avail = batch["availability"].to(self.device, non_blocking=True)
            quality = batch["quality"].to(self.device, non_blocking=True)

            teacher_logits = None
            if consistency_active:
                with torch.no_grad(), torch.autocast(
                    "cuda", enabled=cfg.train.amp and self.device.type == "cuda"
                ):
                    full = torch.ones_like(avail)
                    full_image = batch["full_image"].to(self.device, non_blocking=True)
                    teacher_logits = self.ema.module(full_image, full, quality)

            with torch.autocast("cuda", enabled=cfg.train.amp and self.device.type == "cuda"):
                logits, aux = self.model(image, avail, quality, return_aux=True)
                parts = self.criterion(
                    logits, target, aux.get("deep_logits"), teacher_logits,
                    use_consistency=consistency_active,
                )
                # The final accumulation group may contain fewer microbatches.
                group_start = (i // accum) * accum
                actual_accum = min(accum, len(self.loader) - group_start)
                loss = parts["total"] / actual_accum

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            self.scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(self.loader):
                self._set_lr(self.lr_at(self.state.global_step))
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)

            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            n_batches += 1

        out = {k: v / max(n_batches, 1) for k, v in totals.items()}
        out["epoch_seconds"] = time.perf_counter() - t0
        out["lr"] = self.lr_at(self.state.global_step)
        return out

    # ------------------------------------------------------------------ #
    def fit(self, resume: bool = True) -> TrainState:
        if resume:
            self.maybe_resume()
        cfg = self.cfg
        for epoch in range(self.state.epoch, cfg.train.max_epochs):
            self.state.epoch = epoch
            train_metrics = self.train_epoch(epoch)

            record: Dict[str, Any] = {"epoch": epoch, **train_metrics}
            if (epoch + 1) % cfg.train.val_every == 0 or epoch + 1 == cfg.train.max_epochs:
                val = self.validate_fn(self.model, epoch)
                record.update({f"val_{k}": v for k, v in val.items()})
                score = composite_selection_score(
                    val.get("full_macro_dice", float("nan")),
                    val.get("mean_subset_macro_dice", float("nan")),
                    val.get("worst_subset_macro_dice", float("nan")),
                )
                if not np.isfinite(score):
                    raise FloatingPointError("Checkpoint selection received invalid validation metrics")
                record["selection_score"] = score
                improved = score > self.state.best_score
                if improved:
                    self.state.best_score = score
                    self.state.best_epoch = epoch
                    self.state.epochs_without_improvement = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.state.epochs_without_improvement += 1

            self.state.history.append(record)
            self.log(
                f"epoch {epoch:3d} loss {record.get('total', float('nan')):.4f} "
                f"score {record.get('selection_score', float('nan')):.4f} "
                f"best {self.state.best_score:.4f} @ {self.state.best_epoch} "
                f"({record['epoch_seconds']:.1f}s)"
            )
            # Save every epoch so a Kaggle interruption costs at most one epoch.
            self.save_checkpoint("last.pt")
            self.write_history()

            if self.state.epochs_without_improvement >= cfg.train.early_stopping_patience:
                self.log(
                    f"early stopping at epoch {epoch}: no improvement for "
                    f"{cfg.train.early_stopping_patience} validations"
                )
                break
        return self.state

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, name: str) -> Path:
        """Atomic checkpoint write: temp file then rename."""
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": {
                "epoch": self.state.epoch + 1,
                "global_step": self.state.global_step,
                "best_score": self.state.best_score,
                "best_epoch": self.state.best_epoch,
                "epochs_without_improvement": self.state.epochs_without_improvement,
            },
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash(),
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        if self.train_dataset.quality_norm is not None:
            payload["quality_normalizer"] = self.train_dataset.quality_norm.to_dict()
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def maybe_resume(self, name: str = "last.pt") -> bool:
        path = self.run_dir / name
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("config_hash") != self.cfg.config_hash():
            raise RuntimeError(
                f"Checkpoint {path} was written under a different configuration "
                f"({payload.get('config_hash')} vs {self.cfg.config_hash()}). "
                "Refusing to resume into a different experiment."
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])
        s = payload["state"]
        self.state.epoch = int(s["epoch"])
        self.state.global_step = int(s["global_step"])
        self.state.best_score = float(s["best_score"])
        self.state.best_epoch = int(s["best_epoch"])
        self.state.epochs_without_improvement = int(s["epochs_without_improvement"])
        history_path = self.run_dir / "history.csv"
        if history_path.exists():
            import pandas as pd
            self.state.history = pd.read_csv(history_path).to_dict("records")
        self.log(f"resumed from {path} at epoch {self.state.epoch}")
        return True

    def write_history(self) -> None:
        import pandas as pd
        pd.DataFrame(self.state.history).to_csv(self.run_dir / "history.csv", index=False)


# --------------------------------------------------------------------------- #
def speed_and_memory_pilot(
    cfg: ExperimentConfig, dataset: SliceDataset, steps: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """The 200-step benchmark that plan 11.4 requires before committing to the
    run matrix. Returns measured VRAM and throughput - never planning estimates.
    """
    from .utils import count_parameters, vram_profiler

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    criterion = QMMFLoss(cfg.loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=0,
                        collate_fn=collate)

    model.train()
    it = iter(loader)
    with vram_profiler(0) as stats:
        done = 0
        while done < steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                continue
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            avail = batch["availability"].to(device)
            quality = batch["quality"].to(device)
            with torch.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                logits, aux = model(image, avail, quality, return_aux=True)
                loss = criterion(logits, target, aux.get("deep_logits"))["total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            done += 1

    params = count_parameters(model)
    return {
        "steps": steps,
        "batch_size": cfg.train.batch_size,
        "wall_seconds": stats["wall_seconds"],
        "steps_per_second": steps / max(stats["wall_seconds"], 1e-9),
        "peak_vram_gb": stats["peak_vram_gb"],
        "peak_reserved_gb": stats["peak_reserved_gb"],
        "parameters_total": params["total"],
        "parameters_millions": params["total"] / 1e6,
    }


def check_resource_gates(pilot: Dict[str, float], vram_gate_gb: float = 14.5,
                         param_gate_m: float = 8.0) -> Dict[str, Any]:
    """Plan 7.7 / 11.5 gates, evaluated against *measured* pilot numbers."""
    vram = pilot.get("peak_vram_gb", float("nan"))
    params = pilot.get("parameters_millions", float("nan"))
    return {
        "vram_gate_gb": vram_gate_gb,
        "measured_peak_vram_gb": vram,
        "vram_pass": bool(np.isfinite(vram) and vram < vram_gate_gb),
        "param_gate_millions": param_gate_m,
        "measured_parameters_millions": params,
        "param_pass": bool(np.isfinite(params) and params < param_gate_m),
        "action_if_failed": (
            "Reduce widths/crop or enable activation checkpointing; do not hide "
            "parameter inflation (plan 7.7)."
        ),
    }
