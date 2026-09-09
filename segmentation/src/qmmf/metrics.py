"""Patient-level segmentation metrics (plan 10.1).

Rules that the plan makes non-negotiable and that are enforced here:
  * every metric is computed per patient on the reconstructed 3D volume, then
    summarised across patients - never pooled over slices;
  * empty-reference cases (typically ET) are handled by an explicit,
    pre-specified rule and reported separately, never silently counted as
    Dice 1.0 or dropped;
  * surface metrics use physical spacing in millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

REGIONS = ("wt", "tc", "et")

# Pre-specified empty-reference rule (plan 10.1 "Empty-region rule").
#   both empty        -> Dice 1.0, but flagged and reported in its own column
#   reference empty,
#   prediction non-empty -> Dice 0.0, and counted as a false-positive case
EMPTY_BOTH_DICE = 1.0
EMPTY_REF_FP_DICE = 0.0


@dataclass
class CaseMetrics:
    case_id: str
    region: str
    dice: float
    hd95: float
    nsd: float
    volume_pred_ml: float
    volume_ref_ml: float
    volume_error_ml: float
    relative_volume_error: float
    reference_empty: bool
    prediction_empty: bool
    lesion_recall: float = float("nan")
    lesion_precision: float = float("nan")
    lesion_f1: float = float("nan")


# --------------------------------------------------------------------------- #
def dice_score(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, bool, bool]:
    p = np.asarray(pred, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    p_empty, r_empty = not p.any(), not r.any()
    if r_empty and p_empty:
        return EMPTY_BOTH_DICE, r_empty, p_empty
    if r_empty and not p_empty:
        return EMPTY_REF_FP_DICE, r_empty, p_empty
    if p_empty and not r_empty:
        return 0.0, r_empty, p_empty
    inter = np.logical_and(p, r).sum()
    return float(2 * inter / (p.sum() + r.sum())), r_empty, p_empty


def _surface_distances(
    pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Distances from each predicted surface voxel to the reference surface and
    vice versa, in millimetres."""
    p = np.asarray(pred, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    if not p.any() or not r.any():
        return np.array([]), np.array([])

    def surface(mask):
        eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
        return mask & ~eroded

    sp, sr = surface(p), surface(r)
    dt_to_ref = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return dt_to_ref[sp], dt_to_pred[sr]


def hausdorff95(pred, ref, spacing) -> float:
    d_pr, d_rp = _surface_distances(pred, ref, spacing)
    if d_pr.size == 0 or d_rp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95)))


def normalized_surface_dice(pred, ref, spacing, tolerance_mm: float = 1.0) -> float:
    """Fraction of surface within `tolerance_mm` of the other surface."""
    d_pr, d_rp = _surface_distances(pred, ref, spacing)
    if d_pr.size == 0 or d_rp.size == 0:
        return float("nan")
    num = (d_pr <= tolerance_mm).sum() + (d_rp <= tolerance_mm).sum()
    return float(num / (d_pr.size + d_rp.size))


def lesion_wise_scores(
    pred: np.ndarray, ref: np.ndarray, min_voxels: int = 50,
    overlap_threshold: float = 0.1,
) -> Dict[str, float]:
    """Connected-component detection scores.

    A reference lesion counts as detected when a predicted component overlaps
    at least `overlap_threshold` of it. Components smaller than `min_voxels` are
    ignored on both sides, which is stated rather than tuned per result.
    """
    p_lab, p_n = ndimage.label(np.asarray(pred, dtype=bool))
    r_lab, r_n = ndimage.label(np.asarray(ref, dtype=bool))
    p_ids = [i for i in range(1, p_n + 1) if (p_lab == i).sum() >= min_voxels]
    r_ids = [i for i in range(1, r_n + 1) if (r_lab == i).sum() >= min_voxels]
    if not r_ids and not p_ids:
        return {"recall": float("nan"), "precision": float("nan"), "f1": float("nan")}
    tp_ref = 0
    for ri in r_ids:
        rmask = r_lab == ri
        overlap = np.logical_and(rmask, p_lab > 0).sum() / rmask.sum()
        tp_ref += int(overlap >= overlap_threshold)
    tp_pred = 0
    for pi in p_ids:
        pmask = p_lab == pi
        overlap = np.logical_and(pmask, r_lab > 0).sum() / pmask.sum()
        tp_pred += int(overlap >= overlap_threshold)
    recall = tp_ref / len(r_ids) if r_ids else float("nan")
    precision = tp_pred / len(p_ids) if p_ids else float("nan")
    if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
        f1 = float("nan") if (np.isnan(recall) or np.isnan(precision)) else 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {"recall": float(recall), "precision": float(precision), "f1": float(f1)}


# --------------------------------------------------------------------------- #
def evaluate_case(
    pred: np.ndarray,             # [3, H, W, D] binary
    ref: np.ndarray,              # [3, H, W, D] binary
    spacing: Sequence[float],
    case_id: str = "",
    nsd_tolerance_mm: float = 1.0,
    lesion_metrics: bool = True,
    surface_metrics: bool = True,
) -> List[CaseMetrics]:
    """All metrics for one patient, one row per nested region."""
    voxel_ml = float(np.prod(np.asarray(spacing, dtype=np.float64))) / 1000.0
    rows: List[CaseMetrics] = []
    for i, region in enumerate(REGIONS):
        p, r = np.asarray(pred[i], dtype=bool), np.asarray(ref[i], dtype=bool)
        d, r_empty, p_empty = dice_score(p, r)
        vp, vr = float(p.sum()) * voxel_ml, float(r.sum()) * voxel_ml
        rel = (vp - vr) / vr if vr > 0 else float("nan")
        hd, nsd = float("nan"), float("nan")
        if surface_metrics:
            dp, dr = _surface_distances(p, r, spacing)
            if dp.size and dr.size:
                hd = float(max(np.percentile(dp, 95), np.percentile(dr, 95)))
                nsd = float(((dp <= nsd_tolerance_mm).sum() + (dr <= nsd_tolerance_mm).sum()) / (dp.size + dr.size))
        row = CaseMetrics(
            case_id=case_id, region=region, dice=d,
            hd95=hd,
            nsd=nsd,
            volume_pred_ml=vp, volume_ref_ml=vr,
            volume_error_ml=vp - vr, relative_volume_error=rel,
            reference_empty=r_empty, prediction_empty=p_empty,
        )
        if lesion_metrics:
            ls = lesion_wise_scores(p, r)
            row.lesion_recall = ls["recall"]
            row.lesion_precision = ls["precision"]
            row.lesion_f1 = ls["f1"]
        rows.append(row)
    return rows


def macro_dice(rows: Sequence[CaseMetrics], exclude_empty_reference: bool = True
               ) -> float:
    """Macro Dice over WT/TC/ET for one patient.

    With exclude_empty_reference=True (the pre-specified primary rule), a region
    whose reference is empty does not contribute, so the headline number is not
    inflated by free 1.0 scores.
    """
    vals = [
        r.dice for r in rows
        if not (exclude_empty_reference and r.reference_empty)
    ]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(
    rows: Sequence[CaseMetrics], exclude_empty_reference: bool = True
) -> Dict[str, float]:
    """Cohort summary: per-region means plus the empty-reference bookkeeping."""
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])
    out: Dict[str, float] = {}
    for region in REGIONS:
        sub = df[df.region == region]
        counted = sub[~sub.reference_empty] if exclude_empty_reference else sub
        out[f"dice_{region}"] = float(counted.dice.mean()) if len(counted) else float("nan")
        out[f"hd95_{region}"] = float(counted.hd95.dropna().mean()) if len(counted) else float("nan")
        out[f"nsd_{region}"] = float(counted.nsd.dropna().mean()) if len(counted) else float("nan")
        out[f"rve_{region}"] = float(counted.relative_volume_error.dropna().mean()) if len(counted) else float("nan")
        out[f"n_empty_reference_{region}"] = int(sub.reference_empty.sum())
        out[f"n_evaluable_dice_{region}"] = int(len(counted))
        out[f"n_undefined_hd95_{region}"] = int(counted.hd95.isna().sum())
        out[f"n_empty_prediction_{region}"] = int(sub.prediction_empty.sum())
        out[f"n_empty_ref_false_positive_{region}"] = int(
            (sub.reference_empty & ~sub.prediction_empty).sum()
        )
    out["macro_dice"] = float(np.nanmean(list(per_case_macro(rows).values())))
    return out


def per_case_macro(rows: Sequence[CaseMetrics]) -> Dict[str, float]:
    """case_id -> macro Dice, the unit of every paired statistical test."""
    by_case: Dict[str, List[CaseMetrics]] = {}
    for r in rows:
        by_case.setdefault(r.case_id, []).append(r)
    return {cid: macro_dice(rs) for cid, rs in by_case.items()}
