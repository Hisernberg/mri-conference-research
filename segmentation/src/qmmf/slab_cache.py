"""Lossless slice-chunk cache: identical training samples with less decompression."""
from pathlib import Path
import hashlib
import numpy as np
from .transforms import sample_slice_from_indices


def build_training_slabs(source, destination, slab_depth=8):
    source, destination = Path(source), Path(destination)
    with np.load(source, allow_pickle=False) as z:
        if int(z["cache_version"]) != 2 or list(z["source_channel_order"]) != [1, 2, 3, 0]:
            raise ValueError("Training slabs require a verified canonical version-2 cache")
        image, target, brain = z["image"], z["target"], z["brain_mask"]
        payload = {"format_version": np.array(1), "source_cache_version": np.array(2),
            "source_channel_order": z["source_channel_order"], "quality_raw": z["quality_raw"],
            "slab_depth": np.array(slab_depth), "image_depth": np.array(image.shape[-1]),
            "brain_slices": np.flatnonzero(brain.any(axis=(0, 1))),
            "tumour_slices": np.flatnonzero((target[0] > 0).any(axis=(0, 1)))}
    for start in range(0, image.shape[-1], slab_depth):
        index = start // slab_depth
        payload[f"image_{index:03d}"] = image[..., start:start + slab_depth]
        payload[f"target_{index:03d}"] = target[..., start:start + slab_depth]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **payload)
    # Compare every voxel after compression, not just a checksum of the writer's input.
    with np.load(temp, allow_pickle=False) as z:
        for start in range(0, image.shape[-1], slab_depth):
            index = start // slab_depth
            if not np.array_equal(z[f"image_{index:03d}"], image[..., start:start + slab_depth]):
                raise ValueError("Image values changed during chunk-cache conversion")
            if not np.array_equal(z[f"target_{index:03d}"], target[..., start:start + slab_depth]):
                raise ValueError("Target values changed during chunk-cache conversion")
    temp.replace(destination)
    return {"case_id": source.stem, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "slab_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size, "slab_bytes": destination.stat().st_size,
        "voxel_equality_verified": True, "image_depth": image.shape[-1]}


class TrainingSlabReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._metadata = {}

    def metadata(self, cid):
        if cid not in self._metadata:
            with np.load(self.directory / f"{cid}.npz", allow_pickle=False) as z:
                if int(z["format_version"]) != 1 or int(z["source_cache_version"]) != 2:
                    raise ValueError("Unrecognized training slab format")
                if list(z["source_channel_order"]) != [1, 2, 3, 0]:
                    raise ValueError("Training slab channel order is incorrect")
                self._metadata[cid] = {key: z[key] for key in ["slab_depth", "image_depth",
                    "brain_slices", "tumour_slices", "quality_raw"]}
        return self._metadata[cid]

    def window(self, cid, zslice, context=5):
        if context < 1 or context % 2 != 1:
            raise ValueError("Context must be a positive odd number")
        meta = self.metadata(cid)
        size, depth = int(meta["slab_depth"]), int(meta["image_depth"])
        if not 0 <= zslice < depth:
            raise ValueError("Slice index outside the volume")
        indices = np.clip(np.arange(zslice - context // 2, zslice + context // 2 + 1), 0, depth - 1)
        with np.load(self.directory / f"{cid}.npz", allow_pickle=False) as z:
            slabs = {int(i): z[f"image_{i:03d}"] for i in np.unique(indices // size)}
            window = np.stack([slabs[int(i // size)][..., i % size] for i in indices], axis=1)
            target = z[f"target_{zslice // size:03d}"][..., zslice % size]
        return window.astype(np.float32), target.astype(np.float32)

    def sample(self, cid, cfg, rng):
        meta = self.metadata(cid)
        zslice = sample_slice_from_indices(int(meta["image_depth"]), meta["brain_slices"],
            meta["tumour_slices"], rng, cfg.p_tumor_window, cfg.p_boundary_window)
        window, target = self.window(cid, zslice, cfg.context_slices)
        return window, target, zslice, meta["quality_raw"]
