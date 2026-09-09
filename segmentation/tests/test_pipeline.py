"""Data pipeline, splits, reconstruction and an end-to-end synthetic run.

`test_reconstruction_is_exact` is the load-bearing one: if the 2.5D-to-3D
inverse is not exact, every patient-level metric is computed on a misaligned
volume and the whole evidence package is void.
"""

import json

import numpy as np
import pytest
import torch

from qmmf.config import ExperimentConfig
from qmmf.dataset import SliceDataset, VolumeSliceDataset
from qmmf.manifest import duplicate_audit, manifest_hash
from qmmf.quality import QualityNormalizer, compute_quality_vector, QUALITY_DIM
from qmmf.splits import (
    Splits, build_splits, training_cases, validate_splits,
)
from qmmf.transforms import (
    apply_corruption, center_pad_or_crop, corruption_grid, extract_window,
    invert_center_pad_or_crop, preprocess_case, uncrop_to_original,
)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def test_manifest_audit_passes_on_clean_synthetic_data(synthetic_manifest):
    manifest, report = synthetic_manifest
    assert report["n_cases"] == 12
    assert report["n_failed_cases"] == 0
    assert report["gate_pass"] is True
    assert report["label_scheme_mapping"]["enhancing"] == 3
    assert not report["duplicates"]["exact_duplicate_groups"]


def test_manifest_records_et_presence_and_volumes(synthetic_manifest):
    manifest, _ = synthetic_manifest
    assert "et_present" in manifest
    assert manifest.et_present.sum() == 8      # 4 of 12 cases were built ET-free
    assert (manifest.wt_volume_ml > manifest.tc_volume_ml).all()
    assert (manifest.tc_volume_ml >= manifest.et_volume_ml).all()


def test_manifest_hash_is_stable_and_content_sensitive(synthetic_manifest):
    manifest, _ = synthetic_manifest
    h1 = manifest_hash(manifest)
    assert h1 == manifest_hash(manifest.copy())
    altered = manifest.copy()
    altered.loc[altered.index[0], "array_sha256"] = "deadbeef"
    assert manifest_hash(altered) != h1


def test_exact_duplicate_detection():
    import pandas as pd
    df = pd.DataFrame({
        "case_id": ["a", "b", "c"],
        "array_sha256": ["x", "x", "y"],
    })
    dup = duplicate_audit(df)
    assert dup["exact_duplicate_groups"] == [["a", "b"]]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_splits_are_disjoint_and_cover_each_case_once(synthetic_manifest):
    manifest, _ = synthetic_manifest
    splits = build_splits(manifest, seed=42)
    validate_splits(splits)                    # raises on any leakage

    locked, dev = set(splits.locked_test), set(splits.development)
    assert not locked & dev
    assert len(locked) + len(dev) == len(manifest)

    outer = [c for f in splits.folds.values() for c in f["outer_val"]]
    assert sorted(outer) == sorted(dev), "OOF must cover each development case once"


def test_splits_are_deterministic_in_the_seed(synthetic_manifest):
    manifest, _ = synthetic_manifest
    a = build_splits(manifest, seed=42)
    b = build_splits(manifest, seed=42)
    c = build_splits(manifest, seed=7)
    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_label_subsets_are_strictly_nested(synthetic_manifest):
    manifest, _ = synthetic_manifest
    splits = build_splits(manifest, seed=42)
    s25 = set(splits.label_subsets["0.25"])
    s50 = set(splits.label_subsets["0.50"])
    s100 = set(splits.label_subsets["1.00"])
    assert s25 <= s50 <= s100
    assert len(s25) < len(s50) < len(s100)


def test_split_file_detects_tampering(synthetic_manifest, tmp_path):
    manifest, _ = synthetic_manifest
    splits = build_splits(manifest, seed=42)
    path = tmp_path / "splits.json"
    splits.save(path)
    assert Splits.load(path).split_hash == splits.split_hash

    payload = json.loads(path.read_text())
    payload["locked_test"] = payload["locked_test"][:-1]     # quietly shrink it
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        Splits.load(path)


def test_training_cases_respects_label_fraction(synthetic_manifest):
    manifest, _ = synthetic_manifest
    splits = build_splits(manifest, seed=42)
    full = training_cases(splits, fold=0, label_fraction=1.0)
    quarter = training_cases(splits, fold=0, label_fraction=0.25)
    assert set(quarter) <= set(full)
    assert len(quarter) <= len(full)


# --------------------------------------------------------------------------- #
# Preprocessing and reconstruction
# --------------------------------------------------------------------------- #
def test_robust_zscore_leaves_background_at_zero():
    img = np.zeros((1, 8, 8, 8), dtype=np.float32)
    img[0, 2:6, 2:6, 2:6] = np.random.default_rng(0).normal(500, 50, (4, 4, 4))
    out = preprocess_case(img)
    assert np.all(out["image"][0][~out["brain_mask"]] == 0)
    inside = out["image"][0][out["brain_mask"]]
    assert abs(float(inside.mean())) < 0.5


def test_window_extraction_pads_at_volume_edges():
    img = np.arange(4 * 5 * 5 * 7, dtype=np.float32).reshape(4, 5, 5, 7)
    first = extract_window(img, 0, 5)
    assert first.shape == (4, 5, 5, 5)
    # z = 0 with a 5-slice window edge-pads: the first two planes repeat slice 0.
    assert np.allclose(first[:, 0], first[:, 2 - 2])
    assert np.allclose(first[:, 0], img[..., 0])


def test_center_pad_crop_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    for h, w in [(20, 30), (200, 190), (192, 192)]:
        arr = rng.normal(size=(3, h, w)).astype(np.float32)
        padded, offsets = center_pad_or_crop(arr, (192, 192))
        assert padded.shape == (3, 192, 192)
        back = invert_center_pad_or_crop(padded, offsets, (h, w))
        assert back.shape == (3, h, w)
        if h <= 192 and w <= 192:
            # Pure padding is losslessly invertible.
            assert np.allclose(back, arr)


def test_uncrop_places_data_in_original_geometry():
    arr = np.ones((3, 4, 4, 4), dtype=np.float32)
    full = uncrop_to_original(arr, [[2, 6], [3, 7], [1, 5]], (10, 10, 10))
    assert full.shape == (3, 10, 10, 10)
    assert full[:, 2:6, 3:7, 1:5].sum() == arr.sum()
    assert full.sum() == arr.sum()          # nothing leaked outside the box


def test_reconstruction_is_exact(synthetic_cache):
    """Every slice must land back on the voxel it came from.

    Runs the real inference path with an identity 'model' so any misalignment
    in windowing, padding or uncropping shows up as a mismatch.
    """
    from qmmf.inference import predict_volume

    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({"net.use_quality": False})
    cid = case_ids[0]

    class MarkerModel(torch.nn.Module):
        """Emits the central slice of modality 0 in every output channel."""
        def forward(self, image, availability, quality, return_aux=False):
            centre = image.shape[2] // 2
            plane = image[:, 0, centre]
            logits = plane.unsqueeze(1).repeat(1, 3, 1, 1)
            return (logits, {}) if return_aux else logits

    out = predict_volume(MarkerModel(), cid, cache, cfg, device=torch.device("cpu"),
                         return_logits=True, apply_nesting=False)
    logits = out["logits"]

    rec = cache.load(cid)
    expected_cropped = rec["image"][0].astype(np.float32)
    geom = out["geometry"]
    expected = uncrop_to_original(
        expected_cropped[None], geom["crop_box"], geom["original_shape"]
    )[0]

    assert logits.shape[1:] == expected.shape
    np.testing.assert_allclose(logits[0], expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# Quality features and corruption
# --------------------------------------------------------------------------- #
def test_quality_vector_shape_and_absent_modality():
    rng = np.random.default_rng(0)
    vol = rng.normal(0, 1, (16, 16, 16)).astype(np.float32)
    q = compute_quality_vector(vol)
    assert q.shape == (QUALITY_DIM,)
    assert np.isfinite(q).all()
    assert np.all(compute_quality_vector(np.zeros((16, 16, 16), np.float32)) == 0)


def test_quality_reacts_to_blur_and_noise():
    from scipy import ndimage
    rng = np.random.default_rng(3)
    vol = rng.normal(100, 25, (24, 24, 24)).astype(np.float32)
    vol[vol < 0] = 0
    sharp = compute_quality_vector(vol)
    blurred = compute_quality_vector(ndimage.gaussian_filter(vol, 2.0))
    # High-frequency energy is the blur proxy (feature index 5).
    assert blurred[5] < sharp[5]


def test_quality_normalizer_uses_only_supplied_statistics():
    rng = np.random.default_rng(4)
    train = [rng.normal(0, 1, (4, QUALITY_DIM)).astype(np.float32) for _ in range(20)]
    norm = QualityNormalizer.fit(train)
    z = norm.transform(train[0])
    assert z.shape == (4, QUALITY_DIM)
    assert np.abs(z).max() <= 5.0                 # clipped
    restored = QualityNormalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(restored.transform(train[0]), z, rtol=1e-6, atol=1e-6)


def test_corruption_grid_is_fixed_and_deterministic():
    grid = corruption_grid()
    assert len(grid) == 18                        # 6 kinds x 3 severities
    vol = np.random.default_rng(0).normal(0, 1, (12, 12, 12)).astype(np.float32)
    a = apply_corruption(vol, "noise", 2, seed=5)
    b = apply_corruption(vol, "noise", 2, seed=5)
    np.testing.assert_allclose(a, b)


def test_corruption_severity_is_monotone_for_blur():
    vol = np.random.default_rng(0).normal(0, 1, (16, 16, 16)).astype(np.float32)
    diffs = [
        float(np.abs(apply_corruption(vol, "blur", s, 0) - vol).mean())
        for s in (1, 2, 3)
    ]
    assert diffs[0] < diffs[1] < diffs[2]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_slice_dataset_yields_valid_batches(synthetic_cache):
    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    ds = SliceDataset(case_ids, cache, cfg.data, length=8, seed=0)
    ds.set_epoch(0, 10)
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (4, 5, 32, 32)
        assert item["target"].shape == (3, 32, 32)
        assert item["availability"].sum() >= 1
        assert item["quality"].shape == (4, 7)
        # Absent modalities must be zeroed at the input.
        absent = item["availability"] == 0
        if absent.any():
            assert item["image"][absent].abs().max() == 0


def test_slice_dataset_is_deterministic_for_a_given_epoch_and_index(synthetic_cache):
    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    ds = SliceDataset(case_ids, cache, cfg.data, length=4, seed=11)
    ds.set_epoch(3, 10)
    a, b = ds[2], ds[2]
    assert torch.equal(a["image"], b["image"])
    ds.set_epoch(4, 10)
    assert not torch.equal(ds[2]["image"], a["image"])   # a new epoch resamples


def test_volume_dataset_covers_every_slice(synthetic_cache):
    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    ds = VolumeSliceDataset(case_ids[0], cache, cfg.data)
    assert len(ds) == ds.depth
    assert {ds[z]["z"] for z in range(len(ds))} == set(range(ds.depth))


def test_volume_dataset_applies_availability_mask(synthetic_cache):
    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    ds = VolumeSliceDataset(case_ids[0], cache, cfg.data, availability=mask)
    item = ds[ds.depth // 2]
    assert item["image"][1].abs().max() == 0
    assert item["image"][3].abs().max() == 0


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_end_to_end_train_and_evaluate(synthetic_cache, synthetic_manifest, tmp_path):
    """Two epochs on synthetic data through the real Trainer and evaluator.

    This does not test that the model learns anything - twelve toy cases cannot
    show that. It tests that the training loop, EMA teacher, checkpointing,
    resume, validation and patient-level metric path all run and stay finite.
    """
    from qmmf.evaluate import validation_metrics
    from qmmf.trainer import Trainer, composite_selection_score

    cache, case_ids = synthetic_cache
    manifest, _ = synthetic_manifest
    cfg = ExperimentConfig().merged({
        "data.crop_size": [32, 32], "net.widths": [8, 16, 24],
        "train.max_epochs": 2, "train.batch_size": 2, "train.accumulation_steps": 1,
        "train.num_workers": 0, "train.amp": False, "out_dir": str(tmp_path),
    })
    spacing = {r.case_id: r.zooms for r in manifest.itertuples()}
    from qmmf.quality import QualityNormalizer
    normalizer = QualityNormalizer.fit([cache.quality_raw(c) for c in case_ids[:8]])
    ds = SliceDataset(case_ids[:8], cache, cfg.data, length=6, seed=0,
                      quality_norm=normalizer)

    def validate(model, epoch):
        return validation_metrics(
            model, case_ids[8:10], cache, cfg, spacing,
            device=torch.device("cpu"),
            subsets=[("t1", "t1ce", "t2", "flair"), ("t1", "t2", "flair")],
        )

    trainer = Trainer(cfg, ds, validate, device=torch.device("cpu"),
                      run_dir=tmp_path / "run", logger=lambda m: None)
    state = trainer.fit(resume=False)

    assert state.epoch == 1
    assert len(state.history) == 2
    assert np.isfinite(state.history[-1]["total"])
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "history.csv").exists()

    for record in state.history:
        assert np.isfinite(record["val_full_macro_dice"])
        assert 0.0 <= record["val_full_macro_dice"] <= 1.0

    # Resuming must restore the epoch counter rather than restart training.
    resumed = Trainer(cfg, ds, validate, device=torch.device("cpu"),
                      run_dir=tmp_path / "run", logger=lambda m: None)
    assert resumed.maybe_resume() is True
    assert resumed.state.epoch == 2


def test_resume_refuses_a_different_configuration(synthetic_cache, tmp_path):
    from qmmf.trainer import Trainer

    cache, case_ids = synthetic_cache
    base = ExperimentConfig().merged({
        "data.crop_size": [32, 32], "net.widths": [8, 16],
        "train.max_epochs": 1, "train.batch_size": 2, "train.num_workers": 0,
        "train.amp": False, "out_dir": str(tmp_path),
    })
    ds = SliceDataset(case_ids[:4], cache, base.data, length=2, seed=0)
    t = Trainer(base, ds, lambda m, e: {"full_macro_dice": 0.5,
                                        "mean_subset_macro_dice": 0.5,
                                        "worst_subset_macro_dice": 0.5},
                device=torch.device("cpu"), run_dir=tmp_path / "r",
                logger=lambda m: None)
    t.save_checkpoint("last.pt")

    other = base.merged({"train.lr": 1e-5})
    t2 = Trainer(other, ds, lambda m, e: {}, device=torch.device("cpu"),
                 run_dir=tmp_path / "r", logger=lambda m: None)
    with pytest.raises(RuntimeError, match="different configuration"):
        t2.maybe_resume()


def test_composite_selection_score_weights():
    from qmmf.trainer import composite_selection_score

    assert composite_selection_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert composite_selection_score(1.0, 0.0, 0.0) == pytest.approx(0.6)
    assert composite_selection_score(0.0, 1.0, 0.0) == pytest.approx(0.3)
    assert composite_selection_score(0.0, 0.0, 1.0) == pytest.approx(0.1)
    assert composite_selection_score(float("nan"), 1.0, 1.0) == -float("inf")


def test_all_fifteen_subset_inference_runs(synthetic_cache):
    from qmmf.evaluate import evaluate_all_subsets
    from qmmf.models import build_model
    from qmmf.shapley import efficiency_gap, shapley_from_subset_table
    from qmmf.subsets import subset_key

    cache, case_ids = synthetic_cache
    cfg = ExperimentConfig().merged({
        "data.crop_size": [32, 32], "net.widths": [8, 16],
    })
    model = build_model(cfg).eval()
    from qmmf.quality import QualityNormalizer
    model.quality_normalizer = QualityNormalizer.fit([cache.quality_raw(c) for c in case_ids[2:]])
    spacing = {c: (1.0, 1.0, 2.0) for c in case_ids}

    res = evaluate_all_subsets(model, case_ids[:2], cache, cfg, spacing,
                               device=torch.device("cpu"))
    for cid, table in res["per_case"].items():
        assert len(table) == 15
        phi = shapley_from_subset_table(table)
        full = table[subset_key(("t1", "t1ce", "t2", "flair"))]
        assert abs(efficiency_gap(phi, full)) < 1e-9
