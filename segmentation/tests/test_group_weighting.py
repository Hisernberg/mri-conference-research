import numpy as np
import pytest
from qmmf.config import ExperimentConfig
from qmmf.dataset import SliceDataset
from qmmf.grouping import group_means, group_quality_medians


def test_repeated_scans_do_not_dominate_primary_metric_or_training_sampling():
    mapping = {"a": "one", **{f"b{i}": "two" for i in range(9)}}
    scores = {cid: float(cid != "a") for cid in mapping}
    assert np.mean(list(group_means(scores, mapping).values())) == .5
    ds = SliceDataset(list(mapping), None, ExperimentConfig().data, case_to_group=mapping)
    rng = np.random.default_rng(25)
    proportion = np.mean([ds.sample_case(rng) == "a" for _ in range(10_000)])
    assert .48 < proportion < .52  # Case-uniform sampling would yield about .1.
    with pytest.raises(ValueError, match="known similarity group"):
        group_means({"unknown": .5}, mapping)


def test_quality_donors_are_from_other_training_groups(synthetic_cache):
    cache, ids = synthetic_cache
    training = ids[:6]
    mapping = {cid: str(i // 3) for i, cid in enumerate(training)}
    ds = SliceDataset(training, cache, ExperimentConfig().data,
                      shuffle_quality=True, case_to_group=mapping)
    for cid, donor in ds.quality_donors.items():
        assert donor in training and mapping[cid] != mapping[donor]
    matrices = group_quality_medians(training, cache, mapping)
    assert len(matrices) == 2
    np.testing.assert_allclose(matrices[0], np.median(np.stack([
        cache.quality_raw(cid) for cid in training[:3]]), axis=0))
