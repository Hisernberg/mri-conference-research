"""Calibration, uncertainty, failure detection and selective segmentation.

Plan 10.2. Every fitting step here uses development data only: the temperature
and any region scaling are cross-fitted on the development pool, and the adult
locked test and the pediatric cohort are never used to choose a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration error
# --------------------------------------------------------------------------- #
def _bin_stats(probs, labels, n_bins, adaptive):
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0:
        return np.array([]), np.array([]), np.array([])
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    conf = np.bincount(idx, weights=p, minlength=len(edges) - 1)
    acc = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    nz = counts > 0
    return conf[nz] / counts[nz], acc[nz] / counts[nz], counts[nz]


def expected_calibration_error(probs, labels, n_bins: int = 20) -> float:
    """ECE: count-weighted mean gap over equal-width bins."""
    conf, acc, counts = _bin_stats(probs, labels, n_bins, adaptive=False)
    if conf.size == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(conf - acc)) / counts.sum())


def average_calibration_error(probs, labels, n_bins: int = 20) -> float:
    """mL1-ACE: unweighted mean gap over non-empty *equal-mass* bins.

    Reported alongside ECE because equal-width bins are dominated by the huge
    background mass in segmentation (Barfoot et al., MICCAI 2024).
    """
    conf, acc, _ = _bin_stats(probs, labels, n_bins, adaptive=True)
    if conf.size == 0:
        return float("nan")
    return float(np.mean(np.abs(conf - acc)))


def maximum_calibration_error(probs, labels, n_bins: int = 20) -> float:
    conf, acc, _ = _bin_stats(probs, labels, n_bins, adaptive=False)
    return float(np.max(np.abs(conf - acc))) if conf.size else float("nan")


def brier_score(probs, labels) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def negative_log_likelihood(probs, labels, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64).ravel()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if p.size else float("nan")


def reliability_diagram_data(probs, labels, n_bins: int = 20) -> Dict[str, np.ndarray]:
    conf, acc, counts = _bin_stats(probs, labels, n_bins, adaptive=False)
    return {"confidence": conf, "accuracy": acc, "count": counts}


def calibration_report(probs, labels, n_bins: int = 20) -> Dict[str, float]:
    return {
        "ece": expected_calibration_error(probs, labels, n_bins),
        "ml1_ace": average_calibration_error(probs, labels, n_bins),
        "mce": maximum_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }


# --------------------------------------------------------------------------- #
# Temperature scaling, fitted on development data only
# --------------------------------------------------------------------------- #
@dataclass
class TemperatureScaler:
    """One temperature per region, fitted by minimising NLL on logits.

    `fit` runs a bounded scalar search rather than gradient descent, so it is
    deterministic and needs no torch.
    """

    temperatures: np.ndarray                 # [R]
    regions: Tuple[str, ...] = ("wt", "tc", "et")

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray,
            regions: Sequence[str] = ("wt", "tc", "et"),
            grid: Optional[np.ndarray] = None) -> "TemperatureScaler":
        """logits/labels: [R, N] arrays of development-set voxel samples."""
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        labels = np.atleast_2d(np.asarray(labels, dtype=np.float64))
        grid = np.geomspace(0.25, 4.0, 61) if grid is None else np.asarray(grid)
        temps = np.ones(logits.shape[0])
        for r in range(logits.shape[0]):
            best, best_nll = 1.0, np.inf
            for t in grid:
                p = 1.0 / (1.0 + np.exp(-logits[r] / t))
                nll = negative_log_likelihood(p, labels[r])
                if nll < best_nll:
                    best, best_nll = float(t), nll
            temps[r] = best
        return cls(temperatures=temps, regions=tuple(regions))

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        t = self.temperatures.reshape((-1,) + (1,) * (np.ndim(logits) - 1))
        return np.asarray(logits, dtype=np.float64) / t

    def transform_probs(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.transform_logits(logits)))

    def to_dict(self) -> Dict:
        return {"temperatures": self.temperatures.tolist(), "regions": list(self.regions)}

    @classmethod
    def from_dict(cls, d: Dict) -> "TemperatureScaler":
        return cls(np.array(d["temperatures"], dtype=np.float64), tuple(d["regions"]))


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def predictive_entropy(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Binary entropy per voxel per region."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ensemble_statistics(member_probs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Three-seed deep ensemble: mean, inter-model variance, mean entropy.

    The plan's uncertainty source is the ensemble; entropy of a single model is
    reported as a comparison, not as the primary signal.
    """
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in member_probs], axis=0)
    mean = stack.mean(axis=0)
    return {
        "mean": mean,
        "variance": stack.var(axis=0),
        "entropy_of_mean": predictive_entropy(mean),
        "mean_entropy": predictive_entropy(stack).mean(axis=0),
        "mutual_information": predictive_entropy(mean) - predictive_entropy(stack).mean(axis=0),
    }


def case_uncertainty_summary(
    probs: np.ndarray, brain_mask: Optional[np.ndarray] = None,
    variance: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Scalar per-patient uncertainty features used for failure prediction."""
    ent = predictive_entropy(probs)
    if brain_mask is not None:
        m = np.asarray(brain_mask, dtype=bool)
        ent_vals = ent[..., m] if ent.ndim == 4 else ent[m]
    else:
        ent_vals = ent
    out = {
        "mean_entropy": float(np.mean(ent_vals)),
        "p95_entropy": float(np.percentile(ent_vals, 95)),
        "candidate_entropy": float(
            np.mean(ent[(probs > 0.1) & (probs < 0.9)]) if np.any((probs > 0.1) & (probs < 0.9)) else 0.0
        ),
    }
    if variance is not None:
        out["mean_ensemble_variance"] = float(np.mean(variance))
    return out


# --------------------------------------------------------------------------- #
# Failure detection and selective segmentation
# --------------------------------------------------------------------------- #
def failure_labels(dice_values: Sequence[float], threshold: float) -> np.ndarray:
    """A case is a 'failure' when its macro Dice falls below a threshold fixed
    on development data (plan 10.2 'Failure definition')."""
    return (np.asarray(dice_values, dtype=np.float64) < threshold).astype(int)


def failure_detection_scores(
    uncertainty: Sequence[float], failures: Sequence[int]
) -> Dict[str, float]:
    """AUROC/AUPRC of ranking cases by uncertainty to predict failure."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    u = np.asarray(uncertainty, dtype=np.float64)
    f = np.asarray(failures, dtype=int)
    if f.sum() == 0 or f.sum() == len(f):
        return {"auroc": float("nan"), "auprc": float("nan"),
                "n_failures": int(f.sum()), "n_cases": int(len(f))}
    return {
        "auroc": float(roc_auc_score(f, u)),
        "auprc": float(average_precision_score(f, u)),
        "n_failures": int(f.sum()),
        "n_cases": int(len(f)),
    }


def risk_coverage_curve(
    scores: Sequence[float], uncertainty: Sequence[float],
    risk: str = "one_minus_dice",
) -> Dict[str, np.ndarray]:
    """Risk as a function of retained coverage, ordering by ascending uncertainty.

    Selective segmentation abstains on the most uncertain cases; a useful
    uncertainty signal makes risk decrease monotonically as coverage falls.
    """
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    order = np.argsort(u, kind="stable")
    s_sorted = s[order]
    losses = 1.0 - s_sorted if risk == "one_minus_dice" else s_sorted
    n = len(s_sorted)
    coverage = np.arange(1, n + 1) / n
    risks = np.cumsum(losses) / np.arange(1, n + 1)
    return {"coverage": coverage, "risk": risks, "sorted_score": s_sorted}


def area_under_risk_coverage(curve: Dict[str, np.ndarray]) -> float:
    return float(np.trapezoid(curve["risk"], curve["coverage"])) \
        if hasattr(np, "trapezoid") else float(np.trapz(curve["risk"], curve["coverage"]))


def selective_performance(
    scores: Sequence[float], uncertainty: Sequence[float],
    coverages: Sequence[float] = (1.0, 0.9, 0.8),
) -> Dict[str, float]:
    """Mean score among the retained cases at each coverage level."""
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    order = np.argsort(u, kind="stable")
    out: Dict[str, float] = {}
    for c in coverages:
        k = max(int(round(c * len(s))), 1)
        out[f"score_at_coverage_{int(round(c * 100))}"] = float(s[order][:k].mean())
    return out
