"""Exact Shapley attribution and the statistical machinery.

The Shapley tests check the three axioms that make the values meaningful
(efficiency, symmetry, null player) rather than only checking that the code
runs. If efficiency fails, every attribution figure in the paper is wrong.
"""

import numpy as np
import pytest

from qmmf.shapley import (
    efficiency_gap, exact_shapley, gate_shapley_mismatch, normalize_positive,
    shapley_weights,
)
from qmmf.stats import (
    bootstrap_ci, cohens_dz, compare_paired, holm_correction, non_inferiority,
    paired_bootstrap_difference, permutation_test,
)
from qmmf.subsets import ALL_SUBSETS, subset_key

MODS = ("t1", "t1ce", "t2", "flair")


# --------------------------------------------------------------------------- #
def test_shapley_weights_sum_to_one_per_player():
    n = 4
    w = shapley_weights(n)
    from math import comb
    total = sum(comb(n - 1, s) * w[s] for s in range(n))
    assert total == pytest.approx(1.0)


def test_efficiency_axiom_on_random_value_function():
    """sum of Shapley values must equal v(full) - v(empty), exactly."""
    rng = np.random.default_rng(0)
    table = {s: float(rng.random()) for s in ALL_SUBSETS}
    phi = exact_shapley(table, MODS, empty_value=0.0)
    gap = efficiency_gap(phi, table[MODS], empty_value=0.0)
    assert abs(gap) < 1e-12, gap


def test_efficiency_holds_with_non_zero_empty_value():
    rng = np.random.default_rng(1)
    table = {s: float(rng.random()) for s in ALL_SUBSETS}
    phi = exact_shapley(table, MODS, empty_value=0.3)
    assert abs(efficiency_gap(phi, table[MODS], empty_value=0.3)) < 1e-12


def test_null_player_gets_zero():
    """A modality that never changes the value must have phi = 0."""
    def value(subset):
        return float(len([m for m in subset if m != "t2"]))
    phi = exact_shapley(value, MODS)
    assert phi["t2"] == pytest.approx(0.0, abs=1e-12)


def test_symmetric_players_get_equal_value():
    def value(subset):
        # t1 and t2 are interchangeable in this value function.
        return float(("t1" in subset) + ("t2" in subset))
    phi = exact_shapley(value, MODS)
    assert phi["t1"] == pytest.approx(phi["t2"])


def test_additive_value_function_gives_exact_marginals():
    weights = {"t1": 0.1, "t1ce": 0.4, "t2": 0.2, "flair": 0.3}

    def value(subset):
        return sum(weights[m] for m in subset)

    phi = exact_shapley(value, MODS)
    for m, w in weights.items():
        assert phi[m] == pytest.approx(w, abs=1e-12)


def test_negative_shapley_value_is_representable():
    """An added modality can genuinely hurt (Qiu et al. 2025); the code must
    not clamp that away."""
    def value(subset):
        return 1.0 - 0.5 * ("t2" in subset)
    phi = exact_shapley(value, MODS)
    assert phi["t2"] < 0


def test_string_keyed_tables_are_accepted():
    rng = np.random.default_rng(2)
    table = {subset_key(s): float(rng.random()) for s in ALL_SUBSETS}
    phi = exact_shapley(table, MODS)
    assert set(phi) == set(MODS)


def test_gate_shapley_mismatch_bounds():
    identical = {"t1": 0.25, "t1ce": 0.25, "t2": 0.25, "flair": 0.25}
    assert gate_shapley_mismatch(identical, identical) == pytest.approx(0.0)
    a = {"t1": 1.0, "t1ce": 0.0, "t2": 0.0, "flair": 0.0}
    b = {"t1": 0.0, "t1ce": 1.0, "t2": 0.0, "flair": 0.0}
    assert gate_shapley_mismatch(a, b) == pytest.approx(1.0)


def test_normalize_positive_handles_all_negative():
    out = normalize_positive({"a": -1.0, "b": -2.0})
    assert sum(out.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(0.8, 0.05, 200)
    point, lo, hi = bootstrap_ci(x, n_boot=2000, seed=5)
    assert lo < point < hi
    assert lo < 0.8 < hi


def test_paired_difference_detects_a_real_shift():
    rng = np.random.default_rng(4)
    base = rng.normal(0.75, 0.06, 120)
    better = base + 0.03
    diff, lo, hi = paired_bootstrap_difference(better, base, n_boot=2000, seed=6)
    assert diff == pytest.approx(0.03, abs=1e-9)
    assert lo > 0


def test_permutation_test_p_is_never_zero():
    rng = np.random.default_rng(5)
    a = rng.normal(1.0, 0.01, 60)
    b = a - 5.0
    p = permutation_test(a, b, n_perm=1000, seed=7)
    assert p > 0
    assert p <= 1


def test_permutation_test_on_identical_inputs_is_large():
    x = np.linspace(0.5, 0.9, 40)
    assert permutation_test(x, x.copy(), n_perm=1000) > 0.5


def test_holm_is_monotone_and_at_least_raw():
    p = [0.001, 0.01, 0.04, 0.2]
    adj = holm_correction(p)
    assert np.all(adj >= np.array(p) - 1e-12)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)
    assert adj[0] == pytest.approx(0.004)


def test_holm_never_exceeds_one():
    assert np.all(holm_correction([0.4, 0.5, 0.6, 0.9]) <= 1.0)


def test_non_inferiority_decision_rule():
    rng = np.random.default_rng(6)
    ref = rng.normal(0.80, 0.05, 150)

    slightly_worse = ref - 0.005            # inside a 1.5 pp margin
    r = non_inferiority(slightly_worse, ref, margin=0.015, n_boot=2000)
    assert r["non_inferior"] and not r["superior"]

    much_worse = ref - 0.05                 # outside the margin
    r2 = non_inferiority(much_worse, ref, margin=0.015, n_boot=2000)
    assert not r2["non_inferior"]


def test_compare_paired_reports_effect_size_and_n():
    rng = np.random.default_rng(7)
    a = rng.normal(0.8, 0.05, 80)
    # A constant offset would give the paired differences zero variance, and
    # d_z would be undefined; real per-patient differences vary.
    b = a - rng.normal(0.02, 0.01, 80)
    res = compare_paired(a, b, name="demo", n_boot=1000)
    assert res.n == 80
    assert res.mean_difference == pytest.approx(np.mean(a - b), abs=1e-9)
    assert res.effect_size > 0
    assert res.name == "demo"


def test_effect_size_is_nan_when_differences_have_no_variance():
    """A constant offset makes d_z undefined; report NaN rather than infinity."""
    a = np.linspace(0.7, 0.9, 30)
    res = compare_paired(a, a - 0.02, n_boot=200)
    assert np.isnan(res.effect_size)


def test_nan_pairs_are_dropped_consistently():
    a = np.array([0.8, np.nan, 0.7, 0.9])
    b = np.array([0.7, 0.5, np.nan, 0.85])
    res = compare_paired(a, b, n_boot=500)
    assert res.n == 2               # only the two complete pairs count


def test_cohens_dz_zero_variance_is_nan_not_inf():
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.4, 0.4, 0.4])
    assert np.isnan(cohens_dz(a, b))
