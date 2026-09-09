"""Append-safe experiment queue and ledger (plan 11.2, Appendix C).

Two processes - one per T4 - claim experiments from a shared JSON queue and
append rows to a shared CSV ledger. Both operations use file locking plus
atomic rename so an interrupted Kaggle session cannot corrupt the record or
cause the same experiment to run twice.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LEDGER_FIELDS: List[str] = [
    "run_id", "status", "gpu", "model", "ablation", "fold", "seed",
    "label_fraction", "stage", "config_hash", "manifest_hash", "split_hash",
    "code_hash", "best_epoch", "best_score", "checkpoint",
    "full_macro_dice", "mean_subset_macro_dice", "worst_subset_macro_dice",
    "parameters", "macs", "peak_vram_gb", "wall_seconds", "seconds_per_volume",
    "failure_message", "timestamp", "git_commit",
]

STATUSES = ("pending", "running", "completed", "failed", "resource_rejected")


@contextmanager
def file_lock(path: Path, timeout: float = 60.0):
    """Exclusive advisory lock on a sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    fh = open(lock_path, "a+")
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Could not lock {lock_path} within {timeout}s")
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


# --------------------------------------------------------------------------- #
@dataclass
class ExperimentEntry:
    run_id: str
    model: str
    ablation: str = "A0"
    fold: int = 0
    seed: int = 42
    label_fraction: float = 1.0
    stage: str = "development"
    config_path: str = ""
    status: str = "pending"
    claimed_by: Optional[int] = None
    claimed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentQueue:
    """A JSON list of experiments that two workers claim atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write([])

    # -------------------------------------------------------------- #
    def _read(self) -> List[Dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(self.path)

    # -------------------------------------------------------------- #
    def add(self, entries: Iterable[ExperimentEntry], skip_existing: bool = True) -> int:
        with file_lock(self.path):
            current = self._read()
            known = {e["run_id"] for e in current}
            added = 0
            for e in entries:
                if skip_existing and e.run_id in known:
                    continue
                current.append(e.to_dict())
                known.add(e.run_id)
                added += 1
            self._write(current)
        return added

    def claim(self, gpu: int, stale_after: float = 6 * 3600) -> Optional[Dict[str, Any]]:
        """Atomically take the next pending experiment.

        A 'running' entry older than `stale_after` is reclaimed, which is what
        makes the queue survive a Kaggle session that was cut off mid-run.
        """
        with file_lock(self.path):
            entries = self._read()
            now = time.time()
            for e in entries:
                stale = (
                    e["status"] == "running"
                    and e.get("claimed_at") is not None
                    and now - float(e["claimed_at"]) > stale_after
                )
                if e["status"] == "pending" or stale:
                    e["status"] = "running"
                    e["claimed_by"] = gpu
                    e["claimed_at"] = now
                    self._write(entries)
                    return dict(e)
            return None

    def release(self, run_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unknown status '{status}'. Have {STATUSES}.")
        with file_lock(self.path):
            entries = self._read()
            for e in entries:
                if e["run_id"] == run_id:
                    e["status"] = status
            self._write(entries)

    def summary(self) -> Dict[str, int]:
        entries = self._read()
        out: Dict[str, int] = {s: 0 for s in STATUSES}
        for e in entries:
            out[e["status"]] = out.get(e["status"], 0) + 1
        out["total"] = len(entries)
        return out

    def pending(self) -> List[Dict[str, Any]]:
        return [e for e in self._read() if e["status"] == "pending"]


# --------------------------------------------------------------------------- #
class Ledger:
    """Append-only CSV of run outcomes. One row per run, written under a lock."""

    def __init__(self, path: str | Path, fields: Optional[List[str]] = None):
        self.path = Path(path)
        self.fields = fields or LEDGER_FIELDS
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fields).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in self.fields}
        clean["timestamp"] = clean.get("timestamp") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        with file_lock(self.path):
            with self.path.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fields).writerow(clean)

    def read(self):
        import pandas as pd
        return pd.read_csv(self.path)

    def completed_run_ids(self) -> set:
        df = self.read()
        if df.empty:
            return set()
        return set(df.loc[df.status == "completed", "run_id"].astype(str))

    def record_failure(self, row: Dict[str, Any], exc: BaseException) -> None:
        """Failures are recorded, not swallowed (plan 9.2: 'report failures,
        OOMs and runtime gates rather than silently dropping baselines')."""
        row = dict(row)
        row["status"] = "failed"
        row["failure_message"] = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}"
        )
        self.append(row)
