"""Label-free per-sequence quality vector q (plan 7.3).

Seven features per modality, computed from the image alone. They never see the
segmentation mask, and they are normalised using training-fold statistics only.

The shuffled-quality negative control (ablation A17) and the no-quality control
(A5) are the evidence that decides whether the quality claim survives; this
module provides both.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

QUALITY_FEATURES: List[str] = [
    "snr_proxy",            # robust foreground variation / noise support
    "coefficient_variation",
    "entropy",
    "robust_dynamic_range",
    "zero_support_fraction",
    "high_frequency_energy",
    "center_of_mass_shift",
]
QUALITY_DIM = len(QUALITY_FEATURES)


def _robust_stats(values: np.ndarray) -> Dict[str, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return {"median": med, "mad": mad}


def compute_quality_vector(
    volume: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    n_bins: int = 64,
) -> np.ndarray:
    """Raw (un-normalised) 7-feature quality vector for one modality volume.

    volume: 3D array in native intensity units.
    brain_mask: optional boolean support; defaults to the nonzero voxels.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if brain_mask is None:
        brain_mask = vol != 0
    brain_mask = np.asarray(brain_mask, dtype=bool)

    fg = vol[brain_mask]
    out = np.zeros(QUALITY_DIM, dtype=np.float32)
    if fg.size < 10:
        # An absent or degenerate sequence gets a zero vector; the availability
        # mask, not the quality vector, is what excludes it from fusion.
        return out

    stats = _robust_stats(fg)
    bg = vol[~brain_mask]
    noise = float(np.median(np.abs(bg - np.median(bg)))) if bg.size > 10 else 0.0
    noise = max(noise, 1e-6)

    out[0] = stats["mad"] / noise                                   # snr_proxy
    mean = float(fg.mean())
    out[1] = float(fg.std()) / (abs(mean) + 1e-6)                   # coeff. variation

    p_lo, p_hi = np.percentile(fg, [0.5, 99.5])
    if p_hi > p_lo:
        hist, _ = np.histogram(np.clip(fg, p_lo, p_hi), bins=n_bins, range=(p_lo, p_hi))
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        nz = p[p > 0]
        out[2] = float(-(nz * np.log(nz)).sum() / np.log(n_bins))   # normalised entropy
    p05, p95 = np.percentile(fg, [5, 95])
    out[3] = float(p95 - p05) / (stats["mad"] + 1e-6)               # dynamic range

    out[4] = float((np.abs(fg) < 1e-6).mean())                      # zero support

    # Gradient energy as a blur/sharpness proxy, normalised by intensity scale.
    grads = np.gradient(vol.astype(np.float32))
    gmag = np.sqrt(sum(g ** 2 for g in grads))
    out[5] = float(gmag[brain_mask].mean() / (stats["mad"] + 1e-6))

    # Centre-of-mass displacement relative to the support centroid, in voxels,
    # normalised by volume extent: detects large bias fields / intensity asymmetry.
    coords = np.array(np.nonzero(brain_mask), dtype=np.float32)
    if coords.shape[1] > 0:
        support_c = coords.mean(axis=1)
        weights = np.clip(vol[brain_mask], 0, None)
        wsum = float(weights.sum())
        if wsum > 0:
            intensity_c = (coords * weights[None, :]).sum(axis=1) / wsum
            extent = np.array(vol.shape, dtype=np.float32)
            out[6] = float(np.linalg.norm((intensity_c - support_c) / extent))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_quality_matrix(
    volumes: Sequence[np.ndarray], brain_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Quality matrix [M, Q] for one case, one row per modality."""
    return np.stack(
        [compute_quality_vector(v, brain_mask) for v in volumes], axis=0
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
@dataclass
class QualityNormalizer:
    """Per-modality robust normaliser fitted on training-fold cases only."""

    median: np.ndarray          # [M, Q]
    iqr: np.ndarray             # [M, Q]
    feature_names: List[str]

    @classmethod
    def fit(cls, matrices: Sequence[np.ndarray]) -> "QualityNormalizer":
        stack = np.stack(matrices, axis=0)                    # [N, M, Q]
        med = np.median(stack, axis=0)
        q75, q25 = np.percentile(stack, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        return cls(median=med, iqr=iqr, feature_names=list(QUALITY_FEATURES))

    def transform(self, matrix: np.ndarray, clip: float = 5.0) -> np.ndarray:
        z = (np.asarray(matrix, dtype=np.float32) - self.median) / self.iqr
        return np.clip(z, -clip, clip).astype(np.float32)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["median"] = self.median.tolist()
        d["iqr"] = self.iqr.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "QualityNormalizer":
        return cls(
            median=np.array(d["median"], dtype=np.float32),
            iqr=np.array(d["iqr"], dtype=np.float32),
            feature_names=list(d.get("feature_names", QUALITY_FEATURES)),
        )


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def shuffle_quality_across_patients(
    quality: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A17 negative control: permute quality vectors across patients in a batch.

    If the model's quality claim is real, this must degrade the relevant
    endpoint. If it does not, the quality contribution claim is dropped
    (gate G3).
    """
    q = np.asarray(quality)
    if q.ndim != 3:
        raise ValueError("Expected a batch of quality matrices [B, M, Q].")
    perm = rng.permutation(q.shape[0])
    return q[perm]


def zero_quality(quality: np.ndarray) -> np.ndarray:
    """A5 control: remove the quality signal entirely (availability gate only)."""
    return np.zeros_like(quality)
