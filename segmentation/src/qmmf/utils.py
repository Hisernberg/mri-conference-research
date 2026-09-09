"""Seeding, hashing, environment capture and resource profiling."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch. Deterministic mode is logged, not silently on."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # torch-free contexts (Notebook 01 audit cells)
        pass


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seeding so DataLoader workers do not share a stream."""
    base = int(os.environ.get("QMMF_BASE_SEED", "42"))
    seed = base + worker_id
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash of the decompressed voxel content (catches identical images under
    different filenames, plan 5.5 'exact duplicates')."""
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def affine_hash(affine: np.ndarray, decimals: int = 4) -> str:
    return sha256_array(np.round(np.asarray(affine, dtype=np.float64), decimals))


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def environment_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001 - environment capture must not fail a run
        info["torch"] = f"unavailable: {exc}"
    for mod in ("nibabel", "monai", "scipy", "pandas", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = "not installed"
    info["git_commit"] = _git_commit()
    return info


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:  # noqa: BLE001
        return "unknown"


def code_hash(src_root: str | Path) -> str:
    """Hash every tracked .py file so the ledger records the exact code state."""
    src_root = Path(src_root)
    files = sorted(p for p in src_root.rglob("*.py") if "__pycache__" not in str(p))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(src_root)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Resource profiling (plan 11.3 / 11.5 gates)
# --------------------------------------------------------------------------- #
@contextmanager
def vram_profiler(device: int = 0):
    """Yield a dict that is filled with peak VRAM in GB on exit."""
    stats: Dict[str, float] = {}
    try:
        import torch
        have_cuda = torch.cuda.is_available()
    except ImportError:
        have_cuda = False
    if have_cuda:
        import torch
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        stats["wall_seconds"] = time.perf_counter() - t0
        if have_cuda:
            import torch
            torch.cuda.synchronize(device)
            stats["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            stats["peak_reserved_gb"] = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )
        else:
            stats["peak_vram_gb"] = float("nan")
            stats["peak_reserved_gb"] = float("nan")


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def estimate_macs(model, sample_inputs: Dict[str, Any]) -> Optional[float]:
    """MACs via torch profiler FLOPs when available; None if it cannot be measured.

    Reported as an estimate, never as a verified hardware measurement.
    """
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(**sample_inputs)
        return counter.get_total_flops() / 2.0  # FLOPs -> MACs
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def chunked(items: Iterable, size: int):
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
