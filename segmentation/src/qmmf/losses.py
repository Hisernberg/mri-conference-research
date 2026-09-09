"""Loss composition of plan 8.1-8.2.

    L_seg   = 0.60 * soft_dice + 0.40 * BCE
    L_total = 1.00 * L_seg
            + 0.10 * L_boundary
            + 0.05 * L_nested
            + 0.20 * L_subset_consistency
            + 0.02 * L_calibration     (optional ablation, not assumed beneficial)

Every auxiliary term has a matching ablation (A10, A12, A13 and the calibration
ablation), so a term that does not earn its place is removed and the negative
result is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


# --------------------------------------------------------------------------- #
# Overlap and voxel terms
# --------------------------------------------------------------------------- #
def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample, per-region soft Dice on sigmoid probabilities.

    Averaged over regions then over the batch, so a large WT does not dominate
    a small ET (plan 5.6 of the sibling plan; same discipline applies here).
    """
    probs = torch.sigmoid(logits.float())
    dims = tuple(range(2, probs.dim()))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    loss = 1.0 - dice                                    # [B, R]
    if weights is not None:
        loss = loss * weights.to(loss.dtype).view(1, -1)
        return loss.sum(dim=1).mean() / weights.sum()
    return loss.mean()


def bce_loss(
    logits: torch.Tensor, target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="mean"
    )


def nested_penalty(logits: torch.Tensor) -> torch.Tensor:
    """Penalise ET outside TC and TC outside WT (plan 8.2).

    L = mean(relu(p_ET - p_TC)) + mean(relu(p_TC - p_WT)); differentiable, and
    the hard projection is applied separately at inference.
    """
    p = torch.sigmoid(logits.float())
    wt, tc, et = p[:, 0], p[:, 1], p[:, 2]
    return F.relu(et - tc).mean() + F.relu(tc - wt).mean()


# --------------------------------------------------------------------------- #
# Boundary term
# --------------------------------------------------------------------------- #
def _boundary_band(target: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Dilation minus erosion of the target: a cheap boundary band.

    This is the low-cost implementation of the two the plan screens; the signed
    distance map variant is the alternative, retained only if it clearly
    improves NSD/HD95 (A13 decides).
    """
    pad = width // 2
    dil = F.max_pool2d(target, width, stride=1, padding=pad)
    ero = -F.max_pool2d(-target, width, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def boundary_loss(
    logits: torch.Tensor, target: torch.Tensor, width: int = 3
) -> torch.Tensor:
    """Weighted BCE restricted to the boundary band of the reference.

    Returns 0 for a batch with no boundary voxels (e.g. all-empty ET), rather
    than a NaN.
    """
    band = _boundary_band(target, width)
    total = band.sum()
    if total < 1:
        return logits.float().sum() * 0.0
    per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_voxel * band).sum() / total


# --------------------------------------------------------------------------- #
# Subset consistency with an EMA full-modality teacher
# --------------------------------------------------------------------------- #
def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.90,
    symmetric: bool = False,
) -> torch.Tensor:
    """Match the student's subset prediction to the teacher's full-modality one,
    only where the teacher is confident (plan 8.2).

    Confidence masking is what stops the teacher from teaching its own errors
    near uncertain reference boundaries.
    """
    with torch.no_grad():
        t = torch.sigmoid(teacher_logits.float())
        if symmetric:
            mask = torch.ones_like(t)
        else:
            conf = torch.maximum(t, 1 - t)
            mask = (conf >= confidence_threshold).to(t.dtype)
    s = torch.sigmoid(student_logits.float())
    per_voxel = (s - t) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (per_voxel * mask).sum() / denom


# --------------------------------------------------------------------------- #
# Calibration term (optional ablation)
# --------------------------------------------------------------------------- #
def ml1_ace_loss(
    logits: torch.Tensor, target: torch.Tensor, n_bins: int = 15,
    soft: bool = True, temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable marginal L1 average calibration error.

    Follows the ACE family (Barfoot et al., MICCAI 2024): probabilities are
    assigned to equal-width bins and the absolute gap between mean confidence
    and empirical frequency is averaged over non-empty bins. `soft=True` uses a
    smooth bin assignment so the term is differentiable; the hard-binned
    version in metrics.py is what gets *reported*.
    """
    probs = torch.sigmoid(logits.float()).reshape(-1)
    tgt = target.reshape(-1).to(probs.dtype)
    centers = torch.linspace(
        1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins,
        device=probs.device, dtype=probs.dtype,
    )
    if soft:
        w = torch.exp(-((probs[:, None] - centers[None, :]) ** 2) / (2 * temperature ** 2))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        idx = torch.clamp((probs * n_bins).long(), 0, n_bins - 1)
        w = F.one_hot(idx, n_bins).to(probs.dtype)
    counts = w.sum(dim=0)
    mean_conf = (w * probs[:, None]).sum(dim=0) / counts.clamp_min(1e-8)
    mean_acc = (w * tgt[:, None]).sum(dim=0) / counts.clamp_min(1e-8)
    nonempty = counts > 0
    if not nonempty.any():
        return probs.sum() * 0.0
    return (mean_conf - mean_acc).abs()[nonempty].mean()


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
class QMMFLoss(nn.Module):
    """The full objective, returning every component for the loss-balance audit."""

    def __init__(self, cfg: Optional[LossConfig] = None,
                 region_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        rw = torch.tensor(region_weights if region_weights else [1.0, 1.0, 1.0])
        self.register_buffer("region_weights", rw)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        deep_logits: Optional[List[torch.Tensor]] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        use_consistency: bool = True,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg
        parts: Dict[str, torch.Tensor] = {}

        parts["dice"] = soft_dice_loss(logits, target, weights=self.region_weights)
        parts["bce"] = bce_loss(logits, target)
        seg = c.dice * parts["dice"] + c.bce * parts["bce"]
        parts["seg"] = seg

        total = c.seg * seg

        if c.boundary > 0:
            parts["boundary"] = boundary_loss(logits, target)
            total = total + c.boundary * parts["boundary"]
        if c.nested > 0:
            parts["nested"] = nested_penalty(logits)
            total = total + c.nested * parts["nested"]
        if c.consistency > 0 and teacher_logits is not None and use_consistency:
            parts["consistency"] = consistency_loss(
                logits, teacher_logits, c.consistency_conf_threshold
            )
            total = total + c.consistency * parts["consistency"]
        if c.calibration > 0:
            parts["calibration"] = ml1_ace_loss(logits, target)
            total = total + c.calibration * parts["calibration"]

        # Deep supervision: reduced-weight segmentation loss on intermediate maps.
        if deep_logits:
            ds_total = logits.float().sum() * 0.0
            for w, dl in zip(c.deep_supervision_weights, deep_logits):
                small = F.interpolate(target, size=dl.shape[-2:], mode="nearest")
                ds_total = ds_total + w * (
                    c.dice * soft_dice_loss(dl, small, weights=self.region_weights)
                    + c.bce * bce_loss(dl, small)
                )
            parts["deep_supervision"] = ds_total
            total = total + ds_total

        parts["total"] = total
        return parts


# --------------------------------------------------------------------------- #
def estimate_region_weights(
    positive_fractions: Sequence[float], cap: float = 4.0
) -> torch.Tensor:
    """Inverse-prevalence region weights, estimated from training data and then
    fixed for the whole run (plan 8.2). Capped so ET cannot dominate the loss.
    """
    f = torch.tensor(list(positive_fractions), dtype=torch.float32).clamp_min(1e-6)
    w = (1.0 / f)
    w = w / w.min()
    return w.clamp(max=cap)


@torch.no_grad()
def loss_gradient_audit(
    model: nn.Module, parts: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Report the magnitude of each loss component so a term that dominates by
    an order of magnitude is caught rather than silently rescaled."""
    return {k: float(v.detach()) for k, v in parts.items()}
