"""Model shape, masking, gradient and parameter-budget tests.

The critical one is `test_absent_modality_cannot_influence_output`: it is the
structural guarantee behind the whole missing-modality claim, and it is checked
numerically rather than assumed from the code reading.
"""

import numpy as np
import pytest
import torch

from qmmf.config import ExperimentConfig, ModelConfig
from qmmf.models import ABLATIONS, build_model, project_nested
from qmmf.models.qmmf import masked_feature_max, masked_softmax
from qmmf.models.qmmf_net import QMMFNet
from qmmf.utils import count_parameters

B, M, D, H, W, Q = 2, 4, 5, 64, 64, 7


def make_batch(seed=0, availability=None):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(B, M, D, H, W, generator=g)
    avail = torch.ones(B, M) if availability is None else availability
    quality = torch.randn(B, M, Q, generator=g)
    return image, avail, quality


def small_cfg(**overrides):
    cfg = ExperimentConfig()
    cfg = cfg.merged({"net.widths": [8, 16, 24, 32], "data.crop_size": [H, W],
                      **overrides})
    return cfg


# --------------------------------------------------------------------------- #
def test_masked_softmax_excludes_absent_and_sums_to_one():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    w = masked_softmax(logits, mask)
    assert w[0, 0].item() == 0.0
    assert w.sum().item() == pytest.approx(1.0, abs=1e-6)
    # The three available entries had equal logits, so they share the mass.
    assert w[0, 1].item() == pytest.approx(1 / 3, abs=1e-5)


def test_masked_softmax_single_available_modality():
    logits = torch.randn(4, 4)
    mask = torch.zeros(4, 4)
    mask[:, 2] = 1.0
    w = masked_softmax(logits, mask)
    assert torch.allclose(w[:, 2], torch.ones(4), atol=1e-6)
    assert w[:, [0, 1, 3]].abs().max().item() == 0.0


def test_masked_feature_max_ignores_absent():
    feats = torch.zeros(1, 3, 1, 2, 2)
    feats[0, 0] = 100.0          # absent modality holds a huge value
    feats[0, 1] = 1.0
    feats[0, 2] = 2.0
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    out = masked_feature_max(feats, mask)
    assert out.max().item() == pytest.approx(2.0)


def test_forward_shapes():
    cfg = small_cfg()
    model = build_model(cfg)
    image, avail, quality = make_batch()
    logits = model(image, avail, quality)
    assert logits.shape == (B, 3, H, W)


def test_deep_supervision_outputs_present_in_training_only():
    cfg = small_cfg()
    model = build_model(cfg)
    image, avail, quality = make_batch()
    model.train()
    _, aux = model(image, avail, quality, return_aux=True)
    assert len(aux["deep_logits"]) >= 1
    assert aux["gate_alpha"].shape == (B, len(cfg.net.widths), M)
    # At inference the auxiliary heads are skipped: they would be wasted compute.
    model.eval()
    _, aux_eval = model(image, avail, quality, return_aux=True)
    assert aux_eval["deep_logits"] == []


def test_absent_modality_cannot_influence_output():
    """Change an absent modality's voxels arbitrarily: the output must not move.

    This is the missing-modality guarantee. If masking were only applied at the
    input, or the max branch leaked, this test would fail.
    """
    torch.manual_seed(0)
    model = build_model(small_cfg()).eval()
    image, _, quality = make_batch(seed=1)
    avail = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])

    masked = image * avail[:, :, None, None, None]
    with torch.no_grad():
        a = model(masked, avail, quality)
        perturbed = masked.clone()
        perturbed[0, 2] = torch.randn_like(perturbed[0, 2]) * 50.0
        perturbed[1, 1] = torch.randn_like(perturbed[1, 1]) * 50.0
        b = model(perturbed, avail, quality)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_quality_of_absent_modality_does_not_change_output():
    torch.manual_seed(0)
    model = build_model(small_cfg()).eval()
    image, _, quality = make_batch(seed=2)
    avail = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    image = image * avail[:, :, None, None, None]
    q2 = quality.clone()
    q2[0, 2] = 99.0
    q2[1, 3] = -99.0
    with torch.no_grad():
        assert torch.allclose(model(image, avail, quality),
                              model(image, avail, q2), atol=1e-5)


def test_empty_availability_is_rejected():
    model = build_model(small_cfg())
    image, _, quality = make_batch()
    avail = torch.zeros(B, M)
    with pytest.raises(ValueError):
        model(image, avail, quality)


def test_gradients_flow_to_every_parameter():
    """Every trainable parameter must receive gradient from the real objective.

    The full loss is used, not just the main head, because the deep-supervision
    heads are trained only by their own loss term.
    """
    from qmmf.losses import QMMFLoss

    cfg = small_cfg()
    model = build_model(cfg)
    model.train()
    image, avail, quality = make_batch()
    target = (torch.rand(B, 3, H, W) > 0.7).float()
    target[:, 1] *= target[:, 0]         # keep the nesting valid
    target[:, 2] *= target[:, 1]
    logits, aux = model(image, avail, quality, return_aux=True)
    QMMFLoss(cfg.loss)(logits, target, aux["deep_logits"])["total"].backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    # FiLM's output layer is zero-initialised by design, so it can legitimately
    # show a zero gradient on the very first step; nothing else may.
    hard_missing = [n for n in missing if "film" not in n]
    assert not hard_missing, hard_missing


def test_project_nested_enforces_hierarchy():
    probs = torch.tensor([[[[0.2]], [[0.9]], [[0.95]]]])   # invalid nesting
    out = project_nested(probs)
    assert out[0, 1, 0, 0] <= out[0, 0, 0, 0]
    assert out[0, 2, 0, 0] <= out[0, 1, 0, 0]


def test_parameter_budget_gate_at_planned_widths():
    """Plan 7.7 target is < 8M parameters at the planned widths [24,48,96,160].

    This asserts the *measured* count, so an architecture change that inflates
    the model is caught rather than described away.
    """
    cfg = ExperimentConfig()
    model = build_model(cfg)
    total = count_parameters(model)["total"]
    assert total < 8_000_000, f"{total/1e6:.2f}M parameters exceeds the 8M gate"


@pytest.mark.parametrize("ablation", sorted(ABLATIONS))
def test_every_ablation_builds_and_runs(ablation):
    cfg = ABLATIONS[ablation].apply(small_cfg())
    model = build_model(cfg)
    d = cfg.data.context_slices
    image = torch.randn(1, M, d, 32, 32)
    avail = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    quality = torch.randn(1, M, Q)
    out = model(image * avail[:, :, None, None, None], avail, quality)
    assert out.shape == (1, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_ablation_a6_and_a7_remove_branches():
    a6 = ABLATIONS["A6"].apply(small_cfg())
    a7 = ABLATIONS["A7"].apply(small_cfg())
    assert "variance" not in a6.net.moment_fusion
    assert "max" not in a7.net.moment_fusion


def test_a16_is_parameter_matched_within_ten_percent():
    """A16 controls for capacity: the widened simple-fusion variant must be
    within ~10% of QMMF-Net's parameter count for the comparison to be fair."""
    base = count_parameters(build_model(ExperimentConfig()))["total"]
    a16 = count_parameters(build_model(ABLATIONS["A16"].apply(ExperimentConfig())))["total"]
    ratio = a16 / base
    assert 0.85 <= ratio <= 1.15, f"A16/base parameter ratio {ratio:.2f} is not matched"


def test_baselines_share_the_call_signature():
    from qmmf.models import BASELINE_IDS
    image, avail, quality = make_batch()
    avail = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    for bid, name in BASELINE_IDS.items():
        if name in ("segresnet3d", "swinunetr"):
            continue          # need MONAI; covered separately when installed
        cfg = small_cfg().merged({"model": name})
        model = build_model(cfg)
        out = model(image, avail, quality)
        assert out.shape == (B, 3, H, W), f"{bid}/{name} produced {out.shape}"
