"""Exact patient-level Shapley modality attribution (plan 10.3).

    phi_m = sum over S subset of M without m
              |S|! (|M|-|S|-1)! / |M|!  *  [ v(S union {m}) - v(S) ]

With |M| = 4 there are only 15 non-empty subsets, and every one of them is
evaluated for every locked-test patient, so the Shapley values are exact - no
Monte Carlo sampling and no approximation error to caveat.

Two conventions must be stated because they change the numbers:
  * v(empty set) is not defined by a model prediction (the network requires at
    least one sequence). We set v(empty) = `empty_value`, default 0.0, which
    means phi_m is the contribution relative to predicting nothing. This is
    declared, not silently assumed, and `efficiency_gap` reports the residual.
  * v is patient-level Dice for the region in question, so Shapley values are
    computed separately for WT, TC, ET and macro Dice.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import MODALITIES
from .subsets import ALL_SUBSETS, subset_key


def shapley_weights(n: int) -> Dict[int, float]:
    """Weight |S|! (n-|S|-1)! / n! for each coalition size |S|."""
    return {
        s: factorial(s) * factorial(n - s - 1) / factorial(n)
        for s in range(n)
    }


def exact_shapley(
    value_fn: Mapping[Tuple[str, ...], float] | Callable[[Tuple[str, ...]], float],
    modalities: Sequence[str] = MODALITIES,
    empty_value: float = 0.0,
) -> Dict[str, float]:
    """Exact Shapley values for every modality.

    value_fn: either a dict keyed by subset tuple, or a callable.
    """
    mods = tuple(modalities)
    n = len(mods)
    weights = shapley_weights(n)

    def v(subset: Tuple[str, ...]) -> float:
        if not subset:
            return empty_value
        if callable(value_fn):
            return float(value_fn(subset))
        if subset in value_fn:
            return float(value_fn[subset])
        # Accept string-keyed tables too.
        return float(value_fn[subset_key(subset)])  # type: ignore[index]

    phi: Dict[str, float] = {}
    for m in mods:
        others = [x for x in mods if x != m]
        total = 0.0
        for size in range(len(others) + 1):
            for combo in combinations(others, size):
                s = tuple(sorted(combo, key=mods.index))
                s_with = tuple(sorted(combo + (m,), key=mods.index))
                total += weights[size] * (v(s_with) - v(s))
        phi[m] = float(total)
    return phi


def efficiency_gap(
    phi: Mapping[str, float],
    full_value: float,
    empty_value: float = 0.0,
) -> float:
    """Shapley efficiency axiom check: sum(phi) must equal v(M) - v(empty).

    Returned as a residual so the notebook can assert it is ~0 rather than
    trusting the implementation.
    """
    return float(sum(phi.values()) - (full_value - empty_value))


# --------------------------------------------------------------------------- #
def shapley_from_subset_table(
    subset_scores: Mapping[str, float],
    modalities: Sequence[str] = MODALITIES,
    empty_value: float = 0.0,
) -> Dict[str, float]:
    """Convenience wrapper taking the '+'-joined subset keys used in results
    files, e.g. {'t1': .., 't1+flair': .., ...}."""
    table = {s: float(subset_scores[subset_key(s)]) for s in ALL_SUBSETS}
    return exact_shapley(table, modalities, empty_value)


def patient_shapley_table(
    per_case_subset_scores: Mapping[str, Mapping[str, float]],
    modalities: Sequence[str] = MODALITIES,
    empty_value: float = 0.0,
):
    """One Shapley row per patient.

    per_case_subset_scores: case_id -> {subset_key: score}
    Returns a DataFrame with columns [case_id, <modality>..., efficiency_gap].
    """
    import pandas as pd
    rows = []
    for case_id, table in per_case_subset_scores.items():
        phi = shapley_from_subset_table(table, modalities, empty_value)
        full = float(table[subset_key(tuple(modalities))])
        row = {"case_id": case_id, **phi,
               "efficiency_gap": efficiency_gap(phi, full, empty_value),
               "full_value": full}
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Gate versus Shapley (plan 10.4)
# --------------------------------------------------------------------------- #
def normalize_positive(values: Mapping[str, float]) -> Dict[str, float]:
    """Normalise the *positive* part of a contribution vector to sum to one.

    Negative Shapley values are real and interesting (Qiu et al. 2025 show an
    added modality can hurt), so they are clipped for the comparison with the
    gate simplex but reported separately.
    """
    pos = {k: max(v, 0.0) for k, v in values.items()}
    total = sum(pos.values())
    if total <= 0:
        n = len(pos)
        return {k: 1.0 / n for k in pos}
    return {k: v / total for k, v in pos.items()}


def gate_shapley_mismatch(
    gate_weights: Mapping[str, float],
    shapley_values: Mapping[str, float],
) -> float:
    """Standardised divergence between learned gates and exact contributions.

    Total-variation distance between the two normalised distributions, in
    [0, 1]. This is the reliability signal tested in RQ5 / C4.
    """
    g = normalize_positive(dict(gate_weights))
    s = normalize_positive(dict(shapley_values))
    keys = sorted(set(g) | set(s))
    return float(0.5 * sum(abs(g.get(k, 0.0) - s.get(k, 0.0)) for k in keys))


def gate_shapley_alignment(
    gate_table, shapley_table, modalities: Sequence[str] = MODALITIES,
) -> Dict[str, float]:
    """Per-patient Spearman correlation between gate weights and Shapley values,
    plus the mean mismatch. Bootstrap CIs are added by stats.bootstrap_ci."""
    from scipy.stats import spearmanr
    import pandas as pd

    gate = pd.DataFrame(gate_table).set_index("case_id")
    shap = pd.DataFrame(shapley_table).set_index("case_id")
    shared = sorted(set(gate.index) & set(shap.index))
    if not shared:
        raise ValueError("No overlapping case ids between gate and Shapley tables.")

    rhos, mismatches = [], []
    for cid in shared:
        g = [float(gate.loc[cid, m]) for m in modalities]
        s = [float(shap.loc[cid, m]) for m in modalities]
        if np.std(g) < 1e-12 or np.std(s) < 1e-12:
            rho = np.nan            # a constant vector has no rank correlation
        else:
            rho = float(spearmanr(g, s).statistic)
        rhos.append(rho)
        mismatches.append(gate_shapley_mismatch(
            dict(zip(modalities, g)), dict(zip(modalities, s))
        ))
    return {
        "mean_spearman": float(np.nanmean(rhos)),
        "median_spearman": float(np.nanmedian(rhos)),
        "mean_mismatch": float(np.mean(mismatches)),
        "n_cases": len(shared),
        "n_undefined_spearman": int(np.sum(np.isnan(rhos))),
        "per_case_spearman": rhos,
        "per_case_mismatch": mismatches,
        "case_ids": shared,
    }


def mismatch_predicts_error(
    mismatch: Sequence[float],
    error: Sequence[float],
    entropy: Optional[Sequence[float]] = None,
    modality_count: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Exploratory regression: does gate-Shapley mismatch explain segmentation
    error beyond entropy and modality count?

    Reported as exploratory evidence (RQ5), never as a causal claim.
    """
    import pandas as pd
    import statsmodels.api as sm  # optional dependency; see requirements.txt

    df = pd.DataFrame({"mismatch": np.asarray(mismatch, dtype=float),
                       "error": np.asarray(error, dtype=float)})
    cols = ["mismatch"]
    if entropy is not None:
        df["entropy"] = np.asarray(entropy, dtype=float)
        cols.append("entropy")
    if modality_count is not None:
        df["n_modalities"] = np.asarray(modality_count, dtype=float)
        cols.append("n_modalities")
    df = df.dropna()
    X = sm.add_constant(df[cols])
    model = sm.OLS(df["error"], X).fit()
    return {
        "coef_mismatch": float(model.params["mismatch"]),
        "pvalue_mismatch": float(model.pvalues["mismatch"]),
        "r_squared": float(model.rsquared),
        "n": int(len(df)),
        "summary": model.summary().as_text(),
    }
