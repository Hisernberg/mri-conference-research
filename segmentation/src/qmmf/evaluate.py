"""End-to-end evaluation orchestration.

Ties inference, metrics, subsets, Shapley, calibration and statistics into the
three things the plan actually asks for:

  * `evaluate_cases`        - full-modality patient metrics for a cohort;
  * `evaluate_all_subsets`  - the exhaustive 15-subset table per patient;
  * `validation_metrics`    - the cheap in-training validation used by the
                              composite checkpoint-selection rule.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .calibration import case_uncertainty_summary
from .config import ExperimentConfig
from .dataset import CaseCache
from .inference import binarize, predict_volume
from .metrics import CaseMetrics, evaluate_case, macro_dice, summarize
from .subsets import ALL_SUBSETS, availability_mask, subset_key


DEFAULT_VALIDATION_SUBSETS = [
    ("t1", "t1ce", "t2", "flair"),
    ("t1", "t2", "flair"),          # T1ce missing: the hard TC/ET case
    ("t1", "t1ce", "t2"),           # FLAIR missing: the hard WT case
    ("t1ce",),
    ("flair",),
]


def evaluate_cases(
    model: torch.nn.Module,
    case_ids: Sequence[str],
    cache: CaseCache,
    cfg: ExperimentConfig,
    spacing_lookup: Dict[str, Sequence[float]],
    availability: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    corrupt: Optional[Dict] = None,
    collect_probs: bool = False,
    detailed_metrics: bool = True,
    progress: Optional[str] = None,
) -> Dict[str, object]:
    """Patient-level metrics for a cohort under one availability mask."""
    rows: List[CaseMetrics] = []
    per_case_macro: Dict[str, float] = {}
    uncertainty: Dict[str, Dict[str, float]] = {}
    gates: Dict[str, np.ndarray] = {}
    probs_out: Dict[str, np.ndarray] = {}

    start = time.monotonic()
    for index, cid in enumerate(case_ids, 1):
        out = predict_volume(model, cid, cache, cfg, availability, corrupt=corrupt,
                             device=device)
        probs = out["probs"]
        ref = out["reference"]
        if ref is None:
            raise ValueError(f"Case {cid} has no reference labels to evaluate against.")
        pred = binarize(probs, threshold)
        spacing = spacing_lookup.get(cid, (1.0, 1.0, 1.0))
        case_rows = evaluate_case(pred, ref, spacing, case_id=cid,
                                  nsd_tolerance_mm=cfg.evaluation.nsd_tolerance_mm,
                                  lesion_metrics=detailed_metrics, surface_metrics=detailed_metrics)
        rows.extend(case_rows)
        per_case_macro[cid] = macro_dice(case_rows)
        if detailed_metrics:
            uncertainty[cid] = case_uncertainty_summary(probs)
        if "gate_alpha" in out:
            gates[cid] = out["gate_alpha"]
        if collect_probs:
            probs_out[cid] = probs.astype(np.float16)
        if progress and (index % 25 == 0 or index == len(case_ids)):
            print(f"[{progress}] evaluated {index}/{len(case_ids)} volumes in {time.monotonic() - start:.1f}s", flush=True)

    return {
        "rows": rows,
        "summary": summarize(rows),
        "per_case_macro": per_case_macro,
        "uncertainty": uncertainty,
        "gates": gates,
        "probs": probs_out,
    }


def evaluate_all_subsets(
    model: torch.nn.Module,
    case_ids: Sequence[str],
    cache: CaseCache,
    cfg: ExperimentConfig,
    spacing_lookup: Dict[str, Sequence[float]],
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    subsets: Optional[Sequence[Sequence[str]]] = None,
    progress: Optional[str] = None,
) -> Dict[str, object]:
    """The exhaustive counterfactual table: every case under every subset.

    Returns per_case[case_id][subset_key] = macro Dice, plus per-region tables,
    which is exactly the input exact Shapley needs.
    """
    subsets = subsets or ALL_SUBSETS
    per_case: Dict[str, Dict[str, float]] = {c: {} for c in case_ids}
    per_case_region: Dict[str, Dict[str, Dict[str, float]]] = {
        c: {} for c in case_ids
    }
    subset_summary: Dict[str, Dict[str, float]] = {}
    gates: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in case_ids}

    start = time.monotonic()
    for subset_index, subset in enumerate(subsets, 1):
        key = subset_key(subset)
        mask = availability_mask(subset, cfg.data.modalities)
        result = evaluate_cases(model, case_ids, cache, cfg, spacing_lookup,
                                availability=mask, threshold=threshold, device=device,
                                detailed_metrics=False)
        subset_summary[key] = result["summary"]
        for cid, value in result["per_case_macro"].items():
            per_case[cid][key] = value
        for row in result["rows"]:
            per_case_region[row.case_id].setdefault(row.region, {})[key] = row.dice
        for cid, g in result["gates"].items():
            gates[cid][key] = g
        if progress:
            print(f"[{progress}] modality subsets {subset_index}/{len(subsets)} complete in {time.monotonic() - start:.1f}s", flush=True)

    mean_over_subsets = {
        c: float(np.nanmean(list(v.values()))) for c, v in per_case.items()
    }
    worst_subset = {
        c: float(np.nanmin(list(v.values()))) for c, v in per_case.items()
    }
    return {
        "per_case": per_case,
        "per_case_region": per_case_region,
        "subset_summary": subset_summary,
        "mean_over_subsets": mean_over_subsets,
        "worst_subset": worst_subset,
        "gates": gates,
    }


def validation_metrics(
    model: torch.nn.Module,
    case_ids: Sequence[str],
    cache: CaseCache,
    cfg: ExperimentConfig,
    spacing_lookup: Dict[str, Sequence[float]],
    device: Optional[torch.device] = None,
    subsets: Sequence[Sequence[str]] = tuple(DEFAULT_VALIDATION_SUBSETS),
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """Cheap validation for checkpoint selection.

    Uses a small fixed set of subsets rather than all 15, because validation
    runs every epoch. The full 15-subset evaluation is a locked-test activity.
    """
    ids = list(case_ids)[:max_cases] if max_cases else list(case_ids)
    scores: Dict[str, float] = {}
    per_subset_case: List[List[float]] = []

    for subset in subsets:
        mask = availability_mask(subset, cfg.data.modalities)
        res = evaluate_cases(model, ids, cache, cfg, spacing_lookup,
                             availability=mask, device=device, detailed_metrics=False)
        vals = [res["per_case_macro"][c] for c in ids]
        scores[f"macro_dice::{subset_key(subset)}"] = float(np.nanmean(vals))
        per_subset_case.append(vals)

    full_key = f"macro_dice::{subset_key(tuple(cfg.data.modalities))}"
    arr = np.array(per_subset_case, dtype=float)          # [n_subsets, n_cases]
    return {
        "full_macro_dice": scores.get(full_key, float("nan")),
        "mean_subset_macro_dice": float(np.nanmean(arr)),
        "worst_subset_macro_dice": float(np.nanmean(np.nanmin(arr, axis=0))),
        **scores,
    }
