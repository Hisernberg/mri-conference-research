"""Regression checks for issues found in the actual supplied notebook."""
import numpy as np
import pytest
import torch

from qmmf.config import ExperimentConfig
from qmmf.dataset import SliceDataset, VolumeSliceDataset, canonical_modality_order
from qmmf.inference import binarize, predict_volume
from qmmf.quality import QualityNormalizer


def test_msd_channels_are_reordered_from_metadata():
    metadata = {"modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}}
    assert canonical_modality_order(metadata) == (1, 2, 3, 0)
    with pytest.raises(ValueError):
        canonical_modality_order({"modality": {"0": "unknown"}})


def test_cache_records_verified_channel_order(synthetic_cache):
    cache, ids = synthetic_cache
    assert cache.load(ids[0])["source_channel_order"].tolist() == [1, 2, 3, 0]
    assert int(cache.load(ids[0])["cache_version"]) == 2


def test_quality_normalizer_reaches_real_inference(synthetic_cache):
    cache, ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [16, 16], "train.amp": False})
    norm = QualityNormalizer.fit([cache.quality_raw(c) for c in ids[2:]])
    captured = []

    class Capture(torch.nn.Module):
        def forward(self, image, availability, quality, return_aux=False):
            captured.append(quality.detach().clone())
            logits = torch.zeros(image.shape[0], 3, *image.shape[-2:])
            return logits, {}

    model = Capture()
    with pytest.raises(ValueError, match="normalizer"):
        predict_volume(model, ids[0], cache, cfg, device=torch.device("cpu"))
    model.quality_normalizer = norm
    out = predict_volume(model, ids[0], cache, cfg, device=torch.device("cpu"))
    expected = norm.transform(cache.quality_raw(ids[0]))
    np.testing.assert_allclose(captured[0][0].numpy(), expected)
    assert np.isfinite(out["probs"]).all()
    # Crop size 16 is smaller than the 24x24 volume: no annotated area is lost.
    assert out["reference"].sum() == cache.load(ids[0])["target"].sum()
    ds = VolumeSliceDataset(ids[0], cache, cfg.data)
    assert ds[0]["image"].shape[-1] >= ds.image.shape[2]


def test_subset_student_retains_full_teacher_image(synthetic_cache):
    cache, ids = synthetic_cache
    cfg = ExperimentConfig().merged({"data.crop_size": [32, 32]})
    ds = SliceDataset(ids[:8], cache, cfg.data, curriculum="uniform", augment=False, length=30)
    found = False
    for i in range(30):
        item = ds[i]; absent = item["availability"] == 0
        if absent.any():
            assert item["image"][absent].abs().sum() == 0
            assert item["full_image"][absent].abs().sum() > 0
            assert torch.equal(item["image"][~absent], item["full_image"][~absent])
            found = True
            break
    assert found


def test_shuffled_quality_control_is_active_and_train_only(synthetic_cache):
    cache, ids = synthetic_cache
    cfg = ExperimentConfig()
    ds = SliceDataset(ids[:6], cache, cfg.data, shuffle_quality=True, augment=False)
    sources = [cache.quality_raw(c) for c in ids[:6]]
    for cid, value in ds._shuffled_quality.items():
        assert not np.array_equal(value, cache.quality_raw(cid))
        assert any(np.array_equal(value, q) for q in sources)


def test_nonfinite_prediction_fails_instead_of_becoming_background():
    with pytest.raises(FloatingPointError):
        binarize(np.full((3, 2, 2, 2), np.nan))


def test_half_precision_missing_modality_entropy_stays_finite():
    from qmmf.models.qmmf import masked_softmax, _entropy
    weights = masked_softmax(torch.tensor([[1., 2., 3., 4.]], dtype=torch.float16),
                              torch.tensor([[0., 0., 1., 0.]], dtype=torch.float16))
    assert torch.isfinite(_entropy(weights)).all()
    assert weights[0, 2] == 1


def test_consistency_and_deep_loss_do_not_overflow_at_real_crop_size():
    from qmmf.losses import consistency_loss, QMMFLoss
    from qmmf.config import LossConfig
    # 4*3*192*192 exceeds fp16's largest finite sum; this failed in the original.
    student = torch.full((4, 3, 192, 192), -2., dtype=torch.float16, requires_grad=True)
    teacher = torch.full_like(student, -4.)
    loss = consistency_loss(student, teacher)
    expected = (torch.sigmoid(torch.tensor(-2.)) - torch.sigmoid(torch.tensor(-4.))) ** 2
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert loss > 0
    cfg = LossConfig(boundary=0, nested=0)
    total = QMMFLoss(cfg)(student, torch.zeros_like(student).float(),
                         [student[:, :, ::2, ::2]], teacher)["total"]
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(student.grad).all()


def test_manifest_csv_preserves_binary_fingerprint_and_duplicate_screen(tmp_path):
    import pandas as pd
    from qmmf.manifest import read_manifest, duplicate_audit
    signature = "0" * 80 + "1" * 256 + "0" * 176
    original = pd.DataFrame({"case_id": ["a", "b"], "fingerprint": [signature, signature],
                             "wt_volume_ml": [10.5, 10.5], "et_present": [1, 1]})
    path = tmp_path / "manifest.csv"
    original.to_csv(path, index=False)
    reloaded = read_manifest(path)
    assert reloaded.fingerprint.tolist() == [signature, signature]
    assert reloaded.wt_volume_ml.tolist() == [10.5, 10.5]
    assert duplicate_audit(reloaded) == duplicate_audit(original)
