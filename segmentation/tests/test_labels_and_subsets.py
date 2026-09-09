"""Label mapping, nesting and modality-subset enumeration.

These are the tests the plan calls 'unique-label tests' and 'exact 15-subset'
checks. A silent label-mapping error would invalidate every downstream number,
so it is tested first.
"""

import numpy as np
import pytest

from qmmf.labels import (
    BRATS_LEGACY, MSD_TASK01, LabelSchemeError, assert_nesting,
    infer_label_scheme, region_volumes_ml, scheme_from_dataset_json,
    to_nested_targets,
)
from qmmf.subsets import (
    ALL_SUBSETS, availability_mask, curriculum_probabilities, mask_to_subset,
    sample_availability, subset_key,
)


# --------------------------------------------------------------------------- #
def test_msd_dataset_json_parsed_correctly():
    labels = {"0": "background", "1": "edema",
              "2": "non-enhancing tumor", "3": "enhancing tumour"}
    scheme = scheme_from_dataset_json(labels)
    assert scheme.edema == 1
    assert scheme.non_enhancing == 2
    assert scheme.enhancing == 3


def test_brats_legacy_json_parsed_correctly():
    labels = {"0": "background", "1": "necrotic and non-enhancing tumor core",
              "2": "peritumoral edema", "4": "GD-enhancing tumor"}
    scheme = scheme_from_dataset_json(labels)
    assert scheme.non_enhancing == 1
    assert scheme.edema == 2
    assert scheme.enhancing == 4


def test_non_enhancing_is_not_mistaken_for_enhancing():
    """'non-enhancing' contains 'enhancing'; the parser must not confuse them."""
    scheme = scheme_from_dataset_json(
        {"0": "background", "1": "edema", "2": "non-enhancing", "3": "enhancing"}
    )
    assert scheme.non_enhancing == 2 and scheme.enhancing == 3


def test_unrecognised_label_description_raises():
    with pytest.raises(LabelSchemeError):
        scheme_from_dataset_json({"0": "background", "1": "blob", "2": "thing",
                                  "3": "other"})


def test_observed_label_outside_scheme_raises():
    """A label 4 under the MSD scheme means the mirror is not what we think."""
    seg = np.array([[0, 1], [2, 4]])
    with pytest.raises(LabelSchemeError):
        to_nested_targets(seg, MSD_TASK01)


def test_ambiguous_observed_labels_raise_rather_than_guess():
    # {0,1,2} is a subset of both known schemes: refuse to pick one.
    with pytest.raises(LabelSchemeError):
        infer_label_scheme(dataset_json=None, observed_labels={0, 1, 2})


def test_nested_targets_msd():
    seg = np.array([0, 1, 2, 3])
    wt, tc, et = to_nested_targets(seg, MSD_TASK01)
    assert wt.tolist() == [0, 1, 1, 1]
    assert tc.tolist() == [0, 0, 1, 1]
    assert et.tolist() == [0, 0, 0, 1]


def test_nested_targets_brats_legacy_differs():
    """The same integers mean different regions under the two schemes; this is
    exactly the hazard the plan flags."""
    seg = np.array([0, 1, 2, 4])
    msd_like = to_nested_targets(seg, BRATS_LEGACY)
    assert msd_like[1].tolist() == [0, 1, 0, 1]      # TC = {1, 4} here
    assert msd_like[2].tolist() == [0, 0, 0, 1]


def test_nesting_invariant_holds_and_violation_detected():
    good = np.stack([np.array([1, 1]), np.array([1, 0]), np.array([1, 0])])
    assert_nesting(good)
    bad = np.stack([np.array([1, 0]), np.array([0, 0]), np.array([0, 1])])
    with pytest.raises(LabelSchemeError):
        assert_nesting(bad)


def test_region_volumes_use_spacing():
    targets = np.zeros((3, 10, 10, 10), dtype=np.uint8)
    targets[0, :5] = 1                    # 500 voxels of WT
    vols = region_volumes_ml(targets, (2.0, 1.0, 1.0))   # 2 mm^3 per voxel
    assert vols["wt"] == pytest.approx(500 * 2.0 / 1000.0)


# --------------------------------------------------------------------------- #
def test_exactly_fifteen_non_empty_subsets():
    assert len(ALL_SUBSETS) == 15
    assert len(set(ALL_SUBSETS)) == 15
    assert sum(1 for s in ALL_SUBSETS if len(s) == 1) == 4
    assert sum(1 for s in ALL_SUBSETS if len(s) == 2) == 6
    assert sum(1 for s in ALL_SUBSETS if len(s) == 3) == 4
    assert sum(1 for s in ALL_SUBSETS if len(s) == 4) == 1


def test_availability_mask_roundtrip():
    for subset in ALL_SUBSETS:
        mask = availability_mask(subset)
        assert mask.sum() == len(subset)
        assert mask_to_subset(mask) == subset


def test_empty_subset_is_rejected():
    with pytest.raises(ValueError):
        availability_mask(())


def test_subset_key_is_canonically_ordered():
    assert subset_key(("flair", "t1")) == "t1+flair"
    assert subset_key(("t2", "t1ce")) == "t1ce+t2"


def test_curriculum_probabilities_are_valid_distributions():
    for curriculum in ("v1", "uniform"):
        for frac in (0.0, 0.1, 0.3, 0.7, 0.95, 1.0):
            p = curriculum_probabilities(frac, curriculum)
            assert p.shape == (15,)
            assert np.all(p >= 0)
            assert p.sum() == pytest.approx(1.0)


def test_warmup_phase_favours_full_modality():
    early = curriculum_probabilities(0.05, "v1")
    late = curriculum_probabilities(0.75, "v1")
    full_idx = ALL_SUBSETS.index(("t1", "t1ce", "t2", "flair"))
    assert early[full_idx] > late[full_idx]


def test_sampled_availability_is_never_empty():
    rng = np.random.default_rng(0)
    for frac in (0.0, 0.5, 1.0):
        masks = sample_availability(rng, 64, frac, "v1")
        assert masks.shape == (64, 4)
        assert np.all(masks.sum(axis=1) >= 1)
