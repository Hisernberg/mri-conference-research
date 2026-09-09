import numpy as np
import pandas as pd
import torch

from classification.study import locked_splits, metrics, model_for


def test_duplicate_groups_cannot_cross_any_partition():
    # Two observations per identity must stay together in both nested splits.
    n = 100
    frame = pd.DataFrame({"image_id": [str(i) for i in range(n)],
                          "group_id": [str(i // 2) for i in range(n)],
                          "y": [(i // 2) % 2 for i in range(n)]})
    splits, manifest = locked_splits(frame)
    seen = []
    for part in splits:
        groups = [set(frame.iloc[indices].group_id) for indices in part.values()]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        seen.extend(part["test"])
    assert sorted(seen) == list(range(n))
    assert len(manifest) == n * 5


def test_multiscale_control_preserves_capacity_but_changes_dilation():
    full, control = model_for("full"), model_for("no_multiscale")
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in control.parameters())
    assert full.stage1.b3.dilation == (3, 3)
    assert control.stage1.b3.dilation == (1, 1)
    assert full.stage1.attn is not None
    assert model_for("no_attention").stage1.attn is None


def test_metrics_use_both_classes_and_positive_class_is_yes():
    r = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.8]))
    assert r["sensitivity"] == 1.0
    assert r["specificity"] == 0.5
    assert r["balanced_accuracy"] == 0.75


def test_every_variant_has_finite_forward_and_gradient():
    from classification.study import VARIANTS
    torch.set_num_threads(2)
    for name in VARIANTS:
        model = model_for(name)
        out = model(torch.randn(2, 1, 64, 64))
        assert out.shape == (2, 2)
        loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
