import numpy as np
import torch
from qmmf.config import ExperimentConfig
from qmmf.dataset import CaseCache, SliceDataset
from qmmf.slab_cache import build_training_slabs, TrainingSlabReader
from qmmf.transforms import extract_window, sample_slice_index, sample_slice_from_indices


def test_slab_windows_match_every_voxel_at_boundaries_and_chunk_crossings(synthetic_cache, tmp_path):
    cache, ids = synthetic_cache
    cid = ids[0]; rec = cache.load(cid)
    record = build_training_slabs(cache.path(cid), tmp_path / f"{cid}.npz", slab_depth=3)
    assert record["voxel_equality_verified"]
    reader = TrainingSlabReader(tmp_path)
    for context in [1, 3, 5, 7]:
        for z in range(rec["image"].shape[-1]):
            window, target = reader.window(cid, z, context)
            np.testing.assert_array_equal(window, extract_window(rec["image"], z, context))
            np.testing.assert_array_equal(target, rec["target"][..., z])


def test_chunk_reader_preserves_sampling_augmentation_and_quality(synthetic_cache, tmp_path):
    cache, ids = synthetic_cache
    ids = ids[:4]
    for cid in ids:
        build_training_slabs(cache.path(cid), tmp_path / f"{cid}.npz", slab_depth=3)
    fast = CaseCache(cache.cache_dir, cache.scheme, training_slabs=tmp_path)
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    mapping = {cid: str(i // 2) for i, cid in enumerate(ids)}
    slow_ds = SliceDataset(ids, cache, cfg.data, case_to_group=mapping, shuffle_quality=True, seed=25)
    fast_ds = SliceDataset(ids, fast, cfg.data, case_to_group=mapping, shuffle_quality=True, seed=25)
    for epoch in [0, 5]:
        slow_ds.set_epoch(epoch, 8); fast_ds.set_epoch(epoch, 8)
        for i in range(12):
            slow, quick = slow_ds[i], fast_ds[i]
            assert slow["case_id"] == quick["case_id"] and slow["z"] == quick["z"]
            for key in ["image", "full_image", "target", "availability", "quality"]:
                assert torch.equal(slow[key], quick[key]), key


def test_cached_slice_sampling_matches_empty_and_nonempty_volume_policy():
    for empty_brain, empty_target in [(False, False), (False, True), (True, True)]:
        brain = np.full((2, 2, 7), not empty_brain)
        target = np.zeros_like(brain); target[..., 2:5] = not empty_target
        rng_a, rng_b = np.random.default_rng(18), np.random.default_rng(18)
        for _ in range(200):
            a = sample_slice_index(target, brain, rng_a)
            b = sample_slice_from_indices(7, np.flatnonzero(brain.any(axis=(0, 1))),
                                         np.flatnonzero(target.any(axis=(0, 1))), rng_b)
            assert a == b
        assert rng_a.random() == rng_b.random()
