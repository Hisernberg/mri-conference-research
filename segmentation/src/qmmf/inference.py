"""2.5D inference and exact 3D reconstruction (plan 6.2 step 13, 11.3).

The central slice of each window is the prediction for that slice; slices are
assembled in order, the pad/crop is inverted, and the brain crop is undone, so
the output lands in the original volume geometry voxel-for-voxel. `test_reconstruction`
in the test suite checks that this round-trip is exact on synthetic data.

No test-time augmentation on the default path (plan 11.3); TTA exists only as a
separately reported optional ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import CaseCache, VolumeSliceDataset, collate
from .models import project_nested_numpy
from .transforms import invert_center_pad_or_crop, uncrop_to_original


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    case_id: str,
    cache: CaseCache,
    cfg: ExperimentConfig,
    availability: Optional[np.ndarray] = None,
    corrupt: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    return_logits: bool = False,
    apply_nesting: bool = True,
) -> Dict[str, np.ndarray]:
    """Run one case through the model and reconstruct a full 3D probability map.

    Returns {'probs': [3, H, W, D] in original volume geometry, 'gate_alpha':
    [S, M] mean gate weights, 'geometry': ...}.
    """
    device = device or next(model.parameters()).device
    model.eval()

    normalizer = getattr(model, "quality_normalizer", None)
    if cfg.model == "qmmf_net" and cfg.net.use_quality and normalizer is None:
        raise ValueError("Quality-conditioned inference requires the training-fitted normalizer")
    ds = VolumeSliceDataset(case_id, cache, cfg.data, availability,
                            quality_norm=normalizer, corrupt=corrupt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate)

    cropped_h, cropped_w = ds.image.shape[1], ds.image.shape[2]
    depth = ds.depth
    n_out = len(cfg.data.outputs)
    slices: List[np.ndarray] = []
    gate_alpha: List[np.ndarray] = []
    offsets: Optional[Tuple[int, int, int, int]] = None

    for batch in loader:
        image = batch["image"].to(device)
        avail = batch["availability"].to(device)
        quality = batch["quality"].to(device)
        with torch.autocast("cuda", enabled=(cfg.train.amp and device.type == "cuda")):
            out = model(image, avail, quality, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, {})
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite inference logits for {case_id}")
        arr = logits.cpu().numpy() if return_logits else torch.sigmoid(logits).cpu().numpy()
        slices.append(arr)
        if "gate_alpha" in aux:
            gate_alpha.append(aux["gate_alpha"].float().cpu().numpy())
        offsets = ds._offsets

    stacked = np.concatenate(slices, axis=0)            # [D, 3, h, w] in model space
    stacked = np.moveaxis(stacked, 0, -1)               # [3, h, w, D]

    # Invert the pad/crop back to the cropped-brain grid, slice by slice.
    restored = np.stack([
        invert_center_pad_or_crop(stacked[..., z], offsets, (cropped_h, cropped_w))
        for z in range(depth)
    ], axis=-1)                                         # [3, H_c, W_c, D]

    if apply_nesting and not return_logits:
        restored = project_nested_numpy(restored)

    geom = ds.geometry
    full = uncrop_to_original(restored, geom["crop_box"], geom["original_shape"])

    out: Dict[str, np.ndarray] = {"probs" if not return_logits else "logits": full}
    if gate_alpha:
        out["gate_alpha"] = np.concatenate(gate_alpha, axis=0).mean(axis=0)   # [S, M]
    out["geometry"] = geom
    out["reference"] = (
        uncrop_to_original(ds.target3d, geom["crop_box"], geom["original_shape"])
        if ds.target3d is not None else None
    )
    return out


def binarize(probs: np.ndarray, threshold: float = 0.5,
             enforce_nesting: bool = True) -> np.ndarray:
    """Threshold then (optionally) enforce ET subset TC subset WT on the masks.

    The threshold is fixed on development data only and never tuned on the
    locked test (plan 6.1, gate G6).
    """
    masks = (np.asarray(probs) >= threshold)
    if not np.isfinite(probs).all():
        raise FloatingPointError("Non-finite probabilities cannot be reported as empty predictions")
    if enforce_nesting:
        wt = masks[0]
        tc = masks[1] & wt
        et = masks[2] & tc
        masks = np.stack([wt, tc, et], axis=0)
    return masks


@torch.no_grad()
def predict_all_subsets(
    model: torch.nn.Module,
    case_id: str,
    cache: CaseCache,
    cfg: ExperimentConfig,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Exhaustive inference over the 15 non-empty modality subsets for one case.

    This is what makes the Shapley values exact rather than sampled.
    """
    from .subsets import ALL_SUBSETS, availability_mask, subset_key

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for subset in ALL_SUBSETS:
        mask = availability_mask(subset, cfg.data.modalities)
        out[subset_key(subset)] = predict_volume(
            model, case_id, cache, cfg, availability=mask,
            device=device, batch_size=batch_size,
        )
    return out


@torch.no_grad()
def ensemble_predict(
    models: Sequence[torch.nn.Module],
    case_id: str,
    cache: CaseCache,
    cfg: ExperimentConfig,
    availability: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Three-seed deep ensemble: mean probability plus inter-model variance."""
    members = [
        predict_volume(m, case_id, cache, cfg, availability, device=device,
                       batch_size=batch_size, apply_nesting=False)["probs"]
        for m in models
    ]
    stack = np.stack(members, axis=0)
    mean = stack.mean(axis=0)
    return {
        "probs": project_nested_numpy(mean),
        "variance": stack.var(axis=0),
        "members": stack,
    }


def compress_probabilities(probs: np.ndarray, brain_mask: Optional[np.ndarray] = None
                           ) -> Dict[str, np.ndarray]:
    """Store only what the analysis needs (plan 11.3): float16 probabilities
    inside the brain support, not full feature maps for every patient."""
    p = np.asarray(probs, dtype=np.float16)
    if brain_mask is None:
        return {"probs_f16": p}
    idx = np.nonzero(brain_mask)
    return {
        "values_f16": p[(slice(None),) + idx].astype(np.float16),
        "index": np.stack(idx).astype(np.int16),
        "shape": np.array(p.shape, dtype=np.int32),
    }


def decompress_probabilities(payload: Dict[str, np.ndarray]) -> np.ndarray:
    if "probs_f16" in payload:
        return payload["probs_f16"].astype(np.float32)
    shape = tuple(int(s) for s in payload["shape"])
    out = np.zeros(shape, dtype=np.float32)
    idx = tuple(payload["index"].astype(int))
    out[(slice(None),) + idx] = payload["values_f16"].astype(np.float32)
    return out
