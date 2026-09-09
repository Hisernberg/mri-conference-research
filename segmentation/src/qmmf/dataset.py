"""Case cache and 2.5D torch Datasets.

Two datasets:
  * `SliceDataset`      - random 2.5D windows for training, with the modality
                          curriculum and augmentation;
  * `VolumeSliceDataset` - every slice of one volume in order, for deterministic
                          validation / inference and 3D reconstruction.

Preprocessed cases are cached to compressed .npz once (plan 11.3: "small
persistent cache of preprocessed metadata, not duplicate copies of all volumes
in RAM"). A cache entry stores the cropped, normalised volume in float16 to
keep the Kaggle working directory small.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # audit-only environments
    torch = None

    class Dataset:  # type: ignore
        pass

from .config import MODALITIES, DataConfig
from .labels import LabelScheme, to_nested_targets
from .quality import QualityNormalizer, compute_quality_matrix
from .subsets import availability_mask, sample_subset
from .transforms import (
    AugmentConfig, augment_window, center_pad_or_crop, extract_window,
    preprocess_case, sample_slice_index,
)


def canonical_modality_order(metadata: Dict) -> Tuple[int, ...]:
    """Resolve source indices explicitly; MSD stores FLAIR, T1w, t1gd, T2w."""
    aliases = {"t1": "t1", "t1w": "t1", "t1ce": "t1ce", "t1gd": "t1ce",
               "t1c": "t1ce", "t2": "t2", "t2w": "t2", "flair": "flair"}
    block = metadata.get("modality", metadata.get("channel_names", {}))
    source = {}
    for index, value in block.items():
        name = str(value).lower().replace("-", "").replace("_", "").replace(" ", "")
        if name not in aliases or aliases[name] in source:
            raise ValueError(f"Unknown or repeated modality in metadata: {block}")
        source[aliases[name]] = int(index)
    if set(source) != set(MODALITIES):
        raise ValueError(f"Expected {MODALITIES}, found {block}")
    return tuple(source[m] for m in MODALITIES)


# --------------------------------------------------------------------------- #
# Case cache
# --------------------------------------------------------------------------- #
class CaseCache:
    """Builds and reads the preprocessed per-case cache."""

    def __init__(self, cache_dir: str | Path, scheme: LabelScheme,
                 clip_sigma: float = 5.0, crop_margin: int = 8,
                 max_memory_cases: int = 2, training_slabs: Optional[str | Path] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scheme = scheme
        self.clip_sigma = clip_sigma
        self.crop_margin = crop_margin
        self.max_memory_cases = max_memory_cases
        self._memory = OrderedDict()
        from .slab_cache import TrainingSlabReader
        self.training_slabs = TrainingSlabReader(training_slabs) if training_slabs is not None else None

    def path(self, case_id: str) -> Path:
        return self.cache_dir / f"{case_id}.npz"

    def build(self, case_id: str, image_path: str, label_path: Optional[str],
              overwrite: bool = False) -> Path:
        out = self.path(case_id)
        if out.exists() and not overwrite:
            with np.load(out, allow_pickle=False) as existing:
                if "cache_version" in existing and int(existing["cache_version"]) == 2:
                    return out
        import nibabel as nib

        img = np.asanyarray(nib.load(image_path).dataobj).astype(np.float32)
        if img.ndim != 4:
            raise ValueError(f"{case_id}: expected 4D image, got shape {img.shape}")
        img = np.moveaxis(img, -1, 0)                       # [M, H, W, D]
        metadata_path = next((p / "dataset.json" for p in Path(image_path).parents
                              if (p / "dataset.json").exists()), None)
        if metadata_path is None:
            raise ValueError("dataset.json is required to verify modality order")
        order = canonical_modality_order(json.loads(metadata_path.read_text()))
        img = img[list(order)]
        lbl = (np.asanyarray(nib.load(label_path).dataobj).astype(np.int16)
               if label_path else None)

        prep = preprocess_case(img, lbl, self.clip_sigma, self.crop_margin)
        payload: Dict[str, np.ndarray] = {
            "cache_version": np.array(2, dtype=np.int32),
            "source_channel_order": np.array(order, dtype=np.int32),
            "image": prep["image"].astype(np.float16),
            "brain_mask": prep["brain_mask"].astype(np.bool_),
            "crop_box": np.array(prep["crop_box"], dtype=np.int32),
            "original_shape": np.array(prep["original_shape"], dtype=np.int32),
        }
        # Quality features are computed on the *preprocessed* volume, matching
        # what the network sees, and are stored raw (normalised at load time
        # with training-fold statistics only).
        payload["quality_raw"] = compute_quality_matrix(
            [prep["image"][m] for m in range(prep["image"].shape[0])],
            prep["brain_mask"],
        ).astype(np.float32)

        if lbl is not None:
            if np.count_nonzero(prep["label"]) != np.count_nonzero(lbl):
                raise ValueError(f"{case_id}: brain crop would discard annotated voxels")
            targets = to_nested_targets(prep["label"], self.scheme)
            payload["target"] = targets.astype(np.uint8)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(out)
        self._memory.pop(case_id, None)
        return out

    def load(self, case_id: str) -> Dict[str, np.ndarray]:
        if case_id in self._memory:
            self._memory.move_to_end(case_id)
            return self._memory[case_id]
        with np.load(self.path(case_id), allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
        if int(result.get("cache_version", 0)) != 2:
            raise ValueError("Legacy cache has unverified channel order; rebuild it")
        if self.max_memory_cases:
            self._memory[case_id] = result
            while len(self._memory) > self.max_memory_cases:
                self._memory.popitem(last=False)
        return result

    def quality_raw(self, case_id: str) -> np.ndarray:
        # Reading this one member avoids decompressing an entire volume.
        with np.load(self.path(case_id), allow_pickle=False) as z:
            return z["quality_raw"]

    def sample_training_window(self, case_id, cfg, rng):
        if self.training_slabs is not None:
            return self.training_slabs.sample(case_id, cfg, rng)
        rec = self.load(case_id)
        z = sample_slice_index(rec["target"][0], rec["brain_mask"], rng,
                              p_tumor=cfg.p_tumor_window, p_boundary=cfg.p_boundary_window)
        return (extract_window(rec["image"], z, cfg.context_slices).astype(np.float32),
                rec["target"][..., z].astype(np.float32), z, rec["quality_raw"])


# --------------------------------------------------------------------------- #
def _to_tensor(x: np.ndarray, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.to(dtype) if dtype is not None else t


class SliceDataset(Dataset):
    """Random 2.5D training windows with curriculum modality dropout."""

    def __init__(
        self,
        case_ids: Sequence[str],
        cache: CaseCache,
        data_cfg: DataConfig,
        quality_norm: Optional[QualityNormalizer] = None,
        curriculum: str = "v1",
        augment: bool = True,
        length: Optional[int] = None,
        seed: int = 42,
        shuffle_quality: bool = False,     # ablation A17 negative control
        zero_quality: bool = False,        # ablation A5
        case_to_group: Optional[Dict[str, str]] = None,
    ):
        self.case_ids = list(case_ids)
        if not self.case_ids:
            raise ValueError("SliceDataset received an empty case list.")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Training case IDs must be unique")
        mapping = case_to_group if case_to_group is not None else {c: c for c in self.case_ids}
        if not set(self.case_ids) <= set(mapping):
            raise ValueError("Every training case needs a similarity group")
        self.group_members = {}
        for cid in sorted(self.case_ids):
            self.group_members.setdefault(mapping[cid], []).append(cid)
        self.group_ids = sorted(self.group_members)
        self.cache = cache
        self.cfg = data_cfg
        self.quality_norm = quality_norm
        self.curriculum = curriculum
        self.augment = augment
        self.length = length or (len(self.case_ids) * 32)
        self.seed = seed
        self.shuffle_quality = shuffle_quality
        self.zero_quality = zero_quality
        self.epoch_frac = 0.0
        self._aug = AugmentConfig(
            flip_lr=data_cfg.aug_flip_lr,
            rotation_deg=data_cfg.aug_rotation_deg,
            scale=data_cfg.aug_scale,
            elastic=data_cfg.aug_elastic,
            intensity=data_cfg.aug_intensity,
        )
        self._epoch = 0
        self._shuffled_quality = None
        self.quality_donors = {}
        if shuffle_quality:
            if len(self.group_ids) < 2:
                raise ValueError("Shuffled-quality control needs at least two training groups")
            donor_rng = np.random.default_rng(seed + 17000)
            permutation = donor_rng.permutation(len(self.group_ids))
            donor = np.roll(permutation, 1)
            for i, j in zip(permutation, donor):
                source = self.group_members[self.group_ids[j]]
                for cid in self.group_members[self.group_ids[i]]:
                    self.quality_donors[cid] = source[int(donor_rng.integers(len(source)))]
            self._shuffled_quality = {cid: cache.quality_raw(donor)
                                      for cid, donor in self.quality_donors.items()}

    def set_epoch(self, epoch: int, max_epochs: int) -> None:
        """Advance the curriculum. Called by the trainer once per epoch."""
        self._epoch = epoch
        self.epoch_frac = float(epoch) / max(max_epochs, 1)

    def __len__(self) -> int:
        return self.length

    def sample_case(self, rng) -> str:
        """Equal group probability, then equal probability within that group."""
        group = self.group_ids[int(rng.integers(len(self.group_ids)))]
        members = self.group_members[group]
        return members[int(rng.integers(len(members)))]

    def __getitem__(self, index: int) -> Dict:
        rng = np.random.default_rng((self.seed, self._epoch, index))
        case_id = self.sample_case(rng)
        window, target, z, quality_raw = self.cache.sample_training_window(case_id, self.cfg, rng)

        if self.augment:
            window, target = augment_window(window, target, rng, self._aug)

        window, _ = center_pad_or_crop(window, tuple(self.cfg.crop_size))
        target, _ = center_pad_or_crop(target, tuple(self.cfg.crop_size))

        subset = sample_subset(rng, self.epoch_frac, self.curriculum)
        avail = availability_mask(subset, self.cfg.modalities)

        quality = (self._shuffled_quality[case_id] if self._shuffled_quality is not None
                   else quality_raw).astype(np.float32)
        if self.quality_norm is not None:
            quality = self.quality_norm.transform(quality)
        if self.zero_quality:
            quality = np.zeros_like(quality)

        # Absent modalities are zeroed at the input as well as masked in fusion,
        # so no residual signal can leak through a non-masked path.
        full_window = window
        window = full_window * avail[:, None, None, None]

        return {
            "image": _to_tensor(window, torch.float32),
            "full_image": _to_tensor(full_window, torch.float32),
            "target": _to_tensor(target, torch.float32),
            "availability": _to_tensor(avail, torch.float32),
            "quality": _to_tensor(quality, torch.float32),
            "case_id": case_id,
            "z": int(z),
        }


class VolumeSliceDataset(Dataset):
    """Every slice of one case in order, with a fixed availability mask.

    Used for validation, locked-test inference and every one of the 15 subset
    evaluations. No augmentation, no sampling, no test-time augmentation.
    """

    def __init__(
        self,
        case_id: str,
        cache: CaseCache,
        data_cfg: DataConfig,
        availability: Optional[np.ndarray] = None,
        quality_norm: Optional[QualityNormalizer] = None,
        corrupt: Optional[Dict] = None,
    ):
        self.case_id = case_id
        self.cache = cache
        self.cfg = data_cfg
        self.rec = cache.load(case_id)
        self.image = self.rec["image"].astype(np.float32)
        self.target3d = (self.rec["target"].astype(np.uint8)
                         if "target" in self.rec else None)
        self.depth = self.image.shape[-1]
        # Fully convolutional inference pads to a multiple of 16 and preserves
        # the entire brain field of view; training crops must not truncate test anatomy.
        self.inference_size = tuple(max(int(c), ((int(s) + 15) // 16) * 16)
                                    for c, s in zip(data_cfg.crop_size, self.image.shape[1:3]))
        self.availability = (
            np.ones(len(data_cfg.modalities), dtype=np.float32)
            if availability is None else np.asarray(availability, dtype=np.float32)
        )
        if corrupt is not None:
            self._apply_corruption(corrupt)
        quality = self.rec["quality_raw"].astype(np.float32)
        if corrupt is not None:
            # Quality is recomputed on the corrupted image: the whole point of
            # the quality vector is that it reacts to degraded input.
            quality = compute_quality_matrix(
                [self.image[m] for m in range(self.image.shape[0])],
                self.rec["brain_mask"],
            )
        self.quality = (quality_norm.transform(quality)
                        if quality_norm is not None else quality)
        self.image = self.image * self.availability[:, None, None, None]
        self._offsets: Optional[Tuple[int, int, int, int]] = None

    def _apply_corruption(self, corrupt: Dict) -> None:
        from .transforms import apply_corruption
        targets = corrupt.get("modalities")
        if targets is None:
            targets = [m for m in range(self.image.shape[0]) if self.availability[m] > 0]
        for m in targets:
            if self.availability[m] <= 0:
                continue
            self.image[m] = apply_corruption(
                self.image[m], corrupt["kind"], corrupt["severity"],
                seed=corrupt.get("seed", 0) + m,
            )

    def __len__(self) -> int:
        return self.depth

    def __getitem__(self, z: int) -> Dict:
        window = extract_window(self.image, z, self.cfg.context_slices)
        window, offsets = center_pad_or_crop(window, self.inference_size)
        self._offsets = offsets
        item = {
            "image": _to_tensor(window, torch.float32),
            "availability": _to_tensor(self.availability, torch.float32),
            "quality": _to_tensor(self.quality, torch.float32),
            "z": int(z),
        }
        if self.target3d is not None:
            tgt = self.target3d[..., z].astype(np.float32)
            tgt, _ = center_pad_or_crop(tgt, self.inference_size)
            item["target"] = _to_tensor(tgt, torch.float32)
        return item

    # ------------------------------------------------------------------ #
    @property
    def geometry(self) -> Dict:
        """Everything needed to invert the 2.5D pipeline exactly."""
        return {
            "crop_box": self.rec["crop_box"].tolist(),
            "original_shape": tuple(int(s) for s in self.rec["original_shape"]),
            "cropped_hw": (int(self.image.shape[1]), int(self.image.shape[2])),
            "depth": self.depth,
            "pad_offsets": self._offsets,
        }


def collate(batch: List[Dict]) -> Dict:
    """Default collation that keeps string ids as a list."""
    out: Dict = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if torch is not None and torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out
