"""The 15 non-empty modality subsets and the training curriculum over them.

Plan 9.5: evaluation is exhaustive and exact, not sampled - with |M| = 4 there
are 2^4 - 1 = 15 non-empty subsets, so no Monte Carlo approximation is needed
anywhere in this study (including Shapley, see shapley.py).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import MODALITIES


def all_subsets(modalities: Sequence[str] = MODALITIES) -> List[Tuple[str, ...]]:
    """All 15 non-empty subsets, ordered by size then canonical modality order."""
    out: List[Tuple[str, ...]] = []
    for size in range(1, len(modalities) + 1):
        for combo in combinations(modalities, size):
            out.append(combo)
    return out


def subset_key(subset: Sequence[str]) -> str:
    """Stable identifier, e.g. ('t1','flair') -> 't1+flair'."""
    order = {m: i for i, m in enumerate(MODALITIES)}
    return "+".join(sorted(subset, key=lambda m: order[m]))


def availability_mask(
    subset: Sequence[str], modalities: Sequence[str] = MODALITIES
) -> np.ndarray:
    """Binary availability vector a in {0,1}^M with sum(a) >= 1."""
    mask = np.array([1.0 if m in set(subset) else 0.0 for m in modalities],
                    dtype=np.float32)
    if mask.sum() < 1:
        raise ValueError("The empty modality subset is not a valid input.")
    return mask


def mask_to_subset(
    mask: Sequence[float], modalities: Sequence[str] = MODALITIES
) -> Tuple[str, ...]:
    return tuple(m for m, a in zip(modalities, mask) if a > 0.5)


ALL_SUBSETS: List[Tuple[str, ...]] = all_subsets()
ALL_SUBSET_KEYS: List[str] = [subset_key(s) for s in ALL_SUBSETS]
SUBSETS_BY_SIZE: Dict[int, List[Tuple[str, ...]]] = {
    k: [s for s in ALL_SUBSETS if len(s) == k] for k in (1, 2, 3, 4)
}


# --------------------------------------------------------------------------- #
# Training-time subset curriculum (plan 8.3)
# --------------------------------------------------------------------------- #
# Phase boundaries as fractions of the total epoch budget, with the subset-size
# distribution used in each phase. "hard" phases additionally oversample the
# T1ce-missing and FLAIR-missing subsets, which are the clinically hard cases.
CURRICULUM_V1: List[Dict] = [
    dict(until=0.15, size_probs={4: 0.50, 3: 0.40, 2: 0.10, 1: 0.00}, hard_boost=0.0),
    dict(until=0.60, size_probs={4: 0.30, 3: 0.30, 2: 0.25, 1: 0.15}, hard_boost=0.0),
    dict(until=0.90, size_probs={4: 0.20, 3: 0.25, 2: 0.30, 1: 0.25}, hard_boost=0.5),
    dict(until=1.01, size_probs={4: 0.25, 3: 0.25, 2: 0.25, 1: 0.25}, hard_boost=0.0),
]

# A11 negative control: uniform over the 15 subsets from epoch 1.
CURRICULUM_UNIFORM: List[Dict] = [
    dict(until=1.01, size_probs=None, hard_boost=0.0),
]

CURRICULA = {"v1": CURRICULUM_V1, "uniform": CURRICULUM_UNIFORM}

# Subsets missing the two most informative sequences for TC/ET and WT.
HARD_SUBSETS = [s for s in ALL_SUBSETS if "t1ce" not in s] + \
               [s for s in ALL_SUBSETS if "flair" not in s]


def curriculum_probabilities(
    epoch_frac: float, curriculum: str = "v1"
) -> np.ndarray:
    """Sampling probability over the 15 subsets at a given point in training.

    epoch_frac is (epoch / max_epochs) in [0, 1].
    """
    if curriculum not in CURRICULA:
        raise KeyError(f"Unknown curriculum '{curriculum}'. Have {list(CURRICULA)}.")
    phases = CURRICULA[curriculum]
    phase = next(p for p in phases if epoch_frac < p["until"])

    n = len(ALL_SUBSETS)
    if phase["size_probs"] is None:
        return np.full(n, 1.0 / n, dtype=np.float64)

    probs = np.zeros(n, dtype=np.float64)
    for size, mass in phase["size_probs"].items():
        members = [i for i, s in enumerate(ALL_SUBSETS) if len(s) == size]
        if not members or mass <= 0:
            continue
        for i in members:
            probs[i] += mass / len(members)

    boost = phase.get("hard_boost", 0.0)
    if boost > 0:
        hard_idx = [i for i, s in enumerate(ALL_SUBSETS)
                    if "t1ce" not in s or "flair" not in s]
        probs[hard_idx] *= (1.0 + boost)

    total = probs.sum()
    if total <= 0:
        raise ValueError("Degenerate curriculum phase produced zero mass.")
    return probs / total


def sample_subset(
    rng: np.random.Generator, epoch_frac: float, curriculum: str = "v1"
) -> Tuple[str, ...]:
    probs = curriculum_probabilities(epoch_frac, curriculum)
    idx = rng.choice(len(ALL_SUBSETS), p=probs)
    return ALL_SUBSETS[idx]


def sample_availability(
    rng: np.random.Generator,
    batch_size: int,
    epoch_frac: float,
    curriculum: str = "v1",
) -> np.ndarray:
    """Availability masks [B, M] for one batch. Never returns an all-zero row."""
    probs = curriculum_probabilities(epoch_frac, curriculum)
    idx = rng.choice(len(ALL_SUBSETS), size=batch_size, p=probs)
    return np.stack([availability_mask(ALL_SUBSETS[i]) for i in idx], axis=0)
