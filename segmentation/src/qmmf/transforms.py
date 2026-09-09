"""Preprocessing, 2.5D windowing, augmentation and the fixed corruption suite.

Plan 6.2-6.4 and 9.6. Pure numpy so it runs identically in DataLoader workers.

Two rules are enforced structurally rather than by convention:
  * spatial transforms are applied identically to every modality and to labels
    (nearest-neighbour for labels);
  * the corruption severities used for the locked robustness test are a fixed
    grid that is *not* drawn from the training augmentation distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def brain_support_mask(volume_4d: np.ndarray) -> np.ndarray:
    """Union of nonzero voxels across modalities (plan 6.2 step 8)."""
    return np.any(np.asarray(volume_4d) != 0, axis=0)


def bounding_box(mask: np.ndarray, margin: int = 8) -> Tuple[slice, ...]:
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(coords.min(axis=1) - margin, 0)
    hi = np.minimum(coords.max(axis=1) + 1 + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def robust_zscore(
    volume: np.ndarray, mask: np.ndarray, clip_sigma: float = 5.0
) -> np.ndarray:
    """Per-case, per-modality z-score over nonzero brain, clipped (plan 6.2 step 9).

    Voxels outside the brain support stay at zero so that an absent sequence and
    the background remain distinguishable from normalised tissue.
    """
    vol = np.asarray(volume, dtype=np.float32)
    fg = vol[mask]
    fg = fg[fg != 0]
    if fg.size < 10:
        return np.zeros_like(vol)
    mean, std = float(fg.mean()), float(fg.std())
    if std < 1e-6:
        return np.zeros_like(vol)
    out = np.zeros_like(vol)
    out[mask] = np.clip((vol[mask] - mean) / std, -clip_sigma, clip_sigma)
    return out


def preprocess_case(
    image_4d: np.ndarray,
    label_3d: Optional[np.ndarray] = None,
    clip_sigma: float = 5.0,
    crop_margin: int = 8,
) -> Dict[str, object]:
    """Crop to brain, normalise each modality, and record the crop for
    deterministic reconstruction back into the original volume geometry."""
    image = np.asarray(image_4d, dtype=np.float32)          # [M, H, W, D]
    support = brain_support_mask(image)
    box = bounding_box(support, margin=crop_margin)
    cropped = image[(slice(None),) + box]
    support_c = support[box]
    normed = np.stack(
        [robust_zscore(cropped[m], support_c, clip_sigma) for m in range(cropped.shape[0])],
        axis=0,
    )
    out: Dict[str, object] = {
        "image": normed,
        "brain_mask": support_c,
        "crop_box": [[int(s.start), int(s.stop)] for s in box],
        "original_shape": tuple(int(s) for s in image.shape[1:]),
    }
    if label_3d is not None:
        out["label"] = np.asarray(label_3d)[box]
    return out


def uncrop_to_original(
    array: np.ndarray, crop_box: Sequence[Sequence[int]], original_shape: Sequence[int],
    fill: float = 0.0,
) -> np.ndarray:
    """Inverse of the brain crop; used to place predictions back in volume space."""
    arr = np.asarray(array)
    lead = arr.shape[: arr.ndim - 3]
    full = np.full(tuple(lead) + tuple(original_shape), fill, dtype=arr.dtype)
    sl = (Ellipsis,) + tuple(slice(a, b) for a, b in crop_box)
    full[sl] = arr
    return full


# --------------------------------------------------------------------------- #
# 2.5D windows
# --------------------------------------------------------------------------- #
def extract_window(
    image: np.ndarray, z: int, context: int = 5, axis: int = -1
) -> np.ndarray:
    """Five adjacent axial slices centred on z, edge-padded at the boundaries.

    image: [M, H, W, D]; returns [M, context, H, W].
    """
    if context % 2 != 1:
        raise ValueError("context_slices must be odd so the window has a centre slice.")
    half = context // 2
    depth = image.shape[axis]
    idx = np.clip(np.arange(z - half, z + half + 1), 0, depth - 1)
    win = np.take(image, idx, axis=axis)                # [M, H, W, context]
    return np.moveaxis(win, -1, 1)                      # [M, context, H, W]


def center_pad_or_crop(
    array: np.ndarray, size: Tuple[int, int], value: float = 0.0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Centre pad/crop the last two dims to `size`. Returns the applied offsets
    so the operation can be inverted exactly at reconstruction time."""
    arr = np.asarray(array)
    h, w = arr.shape[-2], arr.shape[-1]
    th, tw = size
    # Crop
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    arr = arr[..., top:top + min(th, h), left:left + min(tw, w)]
    # Pad
    ch, cw = arr.shape[-2], arr.shape[-1]
    pt, pl = (th - ch) // 2, (tw - cw) // 2
    pad = [(0, 0)] * (arr.ndim - 2) + [(pt, th - ch - pt), (pl, tw - cw - pl)]
    arr = np.pad(arr, pad, mode="constant", constant_values=value)
    return arr, (top, left, pt, pl)


def invert_center_pad_or_crop(
    array: np.ndarray, offsets: Tuple[int, int, int, int],
    original_hw: Tuple[int, int], value: float = 0.0,
) -> np.ndarray:
    """Exact inverse of center_pad_or_crop for prediction reconstruction."""
    top, left, pt, pl = offsets
    h, w = original_hw
    arr = np.asarray(array)
    th, tw = arr.shape[-2], arr.shape[-1]
    # Undo the pad, then place the remaining content back on the original canvas.
    ch = min(th - pt, h - top)
    cw = min(tw - pl, w - left)
    inner = arr[..., pt:pt + ch, pl:pl + cw]
    out = np.full(arr.shape[:-2] + (h, w), value, dtype=arr.dtype)
    out[..., top:top + ch, left:left + cw] = inner
    return out


def window_indices(depth: int, stride: int = 1) -> List[int]:
    return list(range(0, depth, stride))


def sample_slice_index(
    label_3d: Optional[np.ndarray],
    brain_mask: np.ndarray,
    rng: np.random.Generator,
    p_tumor: float = 0.50,
    p_boundary: float = 0.25,
) -> int:
    """Slice sampling policy of plan 6.3: 50% tumour, 25% boundary, 25% brain."""
    depth = brain_mask.shape[-1]
    brain_slices = np.nonzero(brain_mask.any(axis=(0, 1)))[0]
    tumour_slices = (np.nonzero((np.asarray(label_3d) > 0).any(axis=(0, 1)))[0]
                     if label_3d is not None else np.array([], dtype=int))
    return sample_slice_from_indices(depth, brain_slices, tumour_slices, rng, p_tumor, p_boundary)


def sample_slice_from_indices(depth, brain_slices, tumour_slices, rng,
                              p_tumor=0.50, p_boundary=0.25):
    """Same RNG sequence as volume-based sampling, using lossless cached indices."""
    if brain_slices.size == 0:
        return int(rng.integers(depth))

    u = rng.random()
    if u < p_tumor and tumour_slices.size:
        return int(rng.choice(tumour_slices))
    if u < p_tumor + p_boundary and tumour_slices.size:
        # Slices at the axial extent of the tumour, where the surface is.
        lo, hi = int(tumour_slices.min()), int(tumour_slices.max())
        band = np.unique(np.clip([lo - 1, lo, lo + 1, hi - 1, hi, hi + 1],
                                 0, depth - 1))
        return int(rng.choice(band))
    return int(rng.choice(brain_slices))


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
@dataclass
class AugmentConfig:
    flip_lr: bool = True
    rotation_deg: float = 10.0
    scale: float = 0.1
    elastic: bool = False
    intensity: bool = True
    p_geometric: float = 0.5
    p_intensity: float = 0.5


def augment_window(
    image: np.ndarray,          # [M, C, H, W]
    target: np.ndarray,         # [3, H, W]
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared spatial transforms for all modalities and labels; independent
    intensity transforms per modality (plan 6.4)."""
    img = np.asarray(image, dtype=np.float32).copy()
    tgt = np.asarray(target).copy()

    if cfg.flip_lr and rng.random() < 0.5:
        img = img[..., ::-1].copy()
        tgt = tgt[..., ::-1].copy()

    if rng.random() < cfg.p_geometric and (cfg.rotation_deg > 0 or cfg.scale > 0):
        angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        zoom = 1.0 + float(rng.uniform(-cfg.scale, cfg.scale))
        img = _affine_2d(img, angle, zoom, order=1)
        tgt = _affine_2d(tgt.astype(np.float32), angle, zoom, order=0).astype(tgt.dtype)

    if cfg.intensity:
        for m in range(img.shape[0]):
            if rng.random() >= cfg.p_intensity:
                continue
            img[m] = _intensity_jitter(img[m], rng)

    return img, tgt


def _affine_2d(array: np.ndarray, angle: float, zoom: float, order: int) -> np.ndarray:
    """Rotate + scale about the image centre, applied to the last two axes."""
    arr = np.asarray(array, dtype=np.float32)
    lead_shape = arr.shape[:-2]
    h, w = arr.shape[-2], arr.shape[-1]
    flat = arr.reshape(-1, h, w)
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta) / zoom, np.sin(theta) / zoom
    matrix = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    centre = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    offset = centre - matrix @ centre
    out = np.stack([
        ndimage.affine_transform(f, matrix, offset=offset, order=order,
                                 mode="constant", cval=0.0)
        for f in flat
    ], axis=0)
    return out.reshape(lead_shape + (h, w))


def _intensity_jitter(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(vol, dtype=np.float32)
    kind = rng.integers(0, 4)
    if kind == 0:                                   # gamma on the positive range
        shift = out.min()
        pos = out - shift
        scale = pos.max() + 1e-6
        gamma = float(rng.uniform(0.7, 1.4))
        out = (pos / scale) ** gamma * scale + shift
    elif kind == 1:                                 # additive Gaussian noise
        out = out + rng.normal(0.0, float(rng.uniform(0.01, 0.10)), out.shape).astype(np.float32)
    elif kind == 2:                                 # contrast / brightness
        out = out * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.1, 0.1))
    else:                                           # mild blur
        out = ndimage.gaussian_filter(out, sigma=(0, *(float(rng.uniform(0.3, 1.0)),) * 2))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Fixed corruption suite (plan 9.6) - locked, not sampled from augmentation
# --------------------------------------------------------------------------- #
CORRUPTIONS: Tuple[str, ...] = (
    "noise", "bias_field", "gamma", "motion", "ghosting", "blur",
)
SEVERITIES: Tuple[int, ...] = (1, 2, 3)

_SEVERITY_TABLE: Dict[str, Dict[int, float]] = {
    "noise":      {1: 0.05, 2: 0.10, 3: 0.20},   # sigma in normalised units
    "bias_field": {1: 0.15, 2: 0.30, 3: 0.50},   # multiplicative field amplitude
    "gamma":      {1: 1.3, 2: 1.6, 3: 2.0},      # gamma exponent
    "motion":     {1: 1.0, 2: 2.0, 3: 4.0},      # ghost displacement in voxels
    "ghosting":   {1: 0.10, 2: 0.20, 3: 0.35},   # ghost amplitude
    "blur":       {1: 0.8, 2: 1.5, 3: 2.5},      # gaussian sigma in voxels
}


def apply_corruption(
    volume: np.ndarray, kind: str, severity: int, seed: int
) -> np.ndarray:
    """Apply one fixed corruption to a single modality. Deterministic in `seed`."""
    if kind not in _SEVERITY_TABLE:
        raise KeyError(f"Unknown corruption '{kind}'. Have {CORRUPTIONS}.")
    if severity not in SEVERITIES:
        raise KeyError(f"Severity must be one of {SEVERITIES}.")
    rng = np.random.default_rng(seed)
    vol = np.asarray(volume, dtype=np.float32).copy()
    level = _SEVERITY_TABLE[kind][severity]

    if kind == "noise":
        # Rician-like magnitude noise on the two quadrature components.
        real = vol + rng.normal(0, level, vol.shape).astype(np.float32)
        imag = rng.normal(0, level, vol.shape).astype(np.float32)
        vol = np.sqrt(real ** 2 + imag ** 2) * np.sign(np.where(vol == 0, 1.0, 1.0))
    elif kind == "bias_field":
        grids = np.meshgrid(*[np.linspace(-1, 1, s) for s in vol.shape], indexing="ij")
        coeffs = rng.normal(0, 1, len(grids) + 1)
        field = coeffs[0] + sum(c * g for c, g in zip(coeffs[1:], grids))
        field = 1.0 + level * (field / (np.abs(field).max() + 1e-6))
        vol = vol * field.astype(np.float32)
    elif kind == "gamma":
        shift = vol.min()
        pos = vol - shift
        scale = pos.max() + 1e-6
        vol = (pos / scale) ** level * scale + shift
    elif kind == "motion":
        shift = int(round(level))
        ghost = np.roll(vol, shift, axis=0)
        vol = 0.75 * vol + 0.25 * ghost
    elif kind == "ghosting":
        ghost = np.roll(vol, vol.shape[1] // 3, axis=1)
        vol = (1 - level) * vol + level * ghost
    elif kind == "blur":
        vol = ndimage.gaussian_filter(vol, sigma=level)
    return vol.astype(np.float32)


def corruption_grid() -> List[Dict[str, object]]:
    """The complete locked corruption grid: 6 kinds x 3 severities."""
    return [{"kind": k, "severity": s} for k in CORRUPTIONS for s in SEVERITIES]
