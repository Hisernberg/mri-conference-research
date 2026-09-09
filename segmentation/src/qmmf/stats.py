"""Patient-level statistical analysis (plan 10.5).

Every function here operates on *paired, patient-level* values. Slice-level
numbers are never valid input; the aggregation happens in metrics.py first.

Reported quantities are effect sizes with confidence intervals; p-values are
secondary and always Holm-corrected within a pre-declared comparison family.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    name: str
    n: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    effect_size: float          # Cohen's d_z for paired data
    p_value: float
    test: str
    p_holm: float = float("nan")

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
    strata: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI, optionally stratified (e.g. by ET presence).

    Returns (point_estimate, ci_low, ci_high).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(statistic(x))

    if strata is None:
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        samples = np.apply_along_axis(statistic, 1, x[idx]) if statistic is not np.mean \
            else x[idx].mean(axis=1)
    else:
        s = np.asarray(strata)[np.isfinite(np.asarray(values, dtype=np.float64))]
        groups = [np.flatnonzero(s == g) for g in np.unique(s)]
        samples = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            pick = np.concatenate([
                g[rng.integers(0, g.size, size=g.size)] for g in groups if g.size
            ])
            samples[b] = statistic(x[pick])
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_difference(
    a: Sequence[float], b: Sequence[float],
    n_boot: int = 10_000, alpha: float = 0.05, seed: int = 12345,
    strata: Optional[Sequence] = None,
) -> Tuple[float, float, float]:
    """CI for mean(a) - mean(b) with patients resampled as pairs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired inputs must have the same length and ordering.")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    st = np.asarray(strata)[keep] if strata is not None else None
    return bootstrap_ci(d, np.mean, n_boot, alpha, seed, st)


def permutation_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10_000, seed: int = 999,
) -> float:
    """Two-sided paired permutation (sign-flip) test on the mean difference."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if d.size == 0:
        return float("nan")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    # +1 in numerator and denominator: the observed assignment is one of the
    # permutations, which keeps the test valid rather than reporting p = 0.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a[keep], b[keep], zero_method="wilcox").pvalue)


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def compare_paired(
    a: Sequence[float], b: Sequence[float], name: str = "",
    test: str = "permutation", n_boot: int = 10_000, alpha: float = 0.05,
    seed: int = 12345, strata: Optional[Sequence] = None,
) -> PairedResult:
    """The standard comparison used everywhere in Notebook 03."""
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a_arr) & np.isfinite(b_arr)
    diff, lo, hi = paired_bootstrap_difference(a, b, n_boot, alpha, seed, strata)
    p = (permutation_test(a, b, seed=seed) if test == "permutation"
         else wilcoxon_test(a, b))
    return PairedResult(
        name=name, n=int(keep.sum()),
        mean_a=float(a_arr[keep].mean()), mean_b=float(b_arr[keep].mean()),
        mean_difference=diff, ci_low=lo, ci_high=hi,
        effect_size=cohens_dz(a, b), p_value=p, test=test,
    )


# --------------------------------------------------------------------------- #
def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, applied *within* a
    pre-declared comparison family (core baselines / confirmed ablations /
    robustness endpoints)."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def apply_holm(results: List[PairedResult]) -> List[PairedResult]:
    adj = holm_correction([r.p_value for r in results])
    for r, a in zip(results, adj):
        r.p_holm = float(a)
    return results


# --------------------------------------------------------------------------- #
def non_inferiority(
    proposed: Sequence[float], reference: Sequence[float],
    margin: float, n_boot: int = 10_000, alpha: float = 0.05,
    seed: int = 4242, strata: Optional[Sequence] = None,
) -> Dict[str, float]:
    """One-sided non-inferiority on the paired mean difference (H1).

    `margin` is expressed in the same units as the scores and must be fixed
    before any locked-test result is viewed (plan 4.1: -1.5 percentage points
    on macro Dice, i.e. margin=0.015 for Dice in [0, 1]).

    Non-inferiority is concluded when the lower bound of the two-sided CI for
    (proposed - reference) lies above -margin.
    """
    diff, lo, hi = paired_bootstrap_difference(
        proposed, reference, n_boot, alpha, seed, strata
    )
    return {
        "mean_difference": diff,
        "ci_low": lo,
        "ci_high": hi,
        "margin": float(margin),
        "non_inferior": bool(np.isfinite(lo) and lo > -abs(margin)),
        "superior": bool(np.isfinite(lo) and lo > 0.0),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------- #
def summarize_subset_performance(
    per_case_subset: Dict[str, Dict[str, float]],
    n_boot: int = 10_000, alpha: float = 0.05, seed: int = 7,
):
    """Mean / worst-subset summary with CIs (endpoints 2 and 3 of plan 4.2).

    per_case_subset: case_id -> {subset_key: macro Dice}
    """
    import pandas as pd
    df = pd.DataFrame(per_case_subset).T          # rows = cases, cols = subsets
    per_case_mean = df.mean(axis=1).to_numpy()
    per_case_worst = df.min(axis=1).to_numpy()
    rows = []
    for label, values in (("mean_over_subsets", per_case_mean),
                          ("worst_subset", per_case_worst)):
        point, lo, hi = bootstrap_ci(values, np.mean, n_boot, alpha, seed)
        rows.append({"endpoint": label, "mean": point, "ci_low": lo, "ci_high": hi,
                     "n": int(np.isfinite(values).sum())})
    for col in df.columns:
        point, lo, hi = bootstrap_ci(df[col].to_numpy(), np.mean, n_boot, alpha, seed)
        rows.append({"endpoint": f"subset::{col}", "mean": point,
                     "ci_low": lo, "ci_high": hi, "n": int(df[col].notna().sum())})
    return pd.DataFrame(rows)


def bootstrap_difference_between_cohorts(
    adult: Sequence[float], pediatric: Sequence[float],
    n_boot: int = 10_000, alpha: float = 0.05, seed: int = 31337,
) -> Dict[str, float]:
    """Unpaired bootstrap for the adult-to-pediatric degradation (plan 10.5).

    The cohorts contain different patients, so this is an unpaired comparison
    and it is reported with the caveat that age, disease and acquisition all
    differ - it is domain-shift evidence, not external validation.
    """
    a = np.asarray(adult, dtype=np.float64)
    p = np.asarray(pediatric, dtype=np.float64)
    a, p = a[np.isfinite(a)], p[np.isfinite(p)]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = (a[rng.integers(0, a.size, a.size)].mean()
                    - p[rng.integers(0, p.size, p.size)].mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "adult_mean": float(a.mean()), "pediatric_mean": float(p.mean()),
        "difference": float(a.mean() - p.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "n_adult": int(a.size), "n_pediatric": int(p.size),
        "caveat": ("Unpaired comparison across cohorts differing in age, tumour "
                   "phenotype, acquisition and label domain; out-of-domain stress "
                   "test, not external validation."),
    }
