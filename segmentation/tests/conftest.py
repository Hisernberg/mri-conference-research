"""Synthetic MSD-like cohort used by the pipeline tests.

Real MSD Task01 volumes are not available in CI, so the fixtures build small
NIfTI cases with the same structure: 4-channel images, integer masks under the
MSD label scheme, valid nesting, and a mix of ET-present and ET-absent cases.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    nib = pytest.importorskip("nibabel")
    root = tmp_path_factory.mktemp("msd_synthetic")
    (root / "imagesTr").mkdir()
    (root / "labelsTr").mkdir()

    rng = np.random.default_rng(0)
    n_cases, shape = 12, (24, 24, 16)

    for i in range(n_cases):
        img = rng.normal(100.0, 20.0, shape + (4,)).astype(np.float32)
        # A zero rim so the brain crop has something to find.
        img[:3], img[-3:], img[:, :3], img[:, -3:] = 0, 0, 0, 0

        seg = np.zeros(shape, dtype=np.uint8)
        cz = 4 + (i % 3)
        seg[6:16, 6:16, cz:cz + 5] = 1                      # edema
        seg[8:13, 8:13, cz + 1:cz + 4] = 2                  # non-enhancing
        if i % 3 != 0:                                       # a third have no ET
            seg[9:12, 9:12, cz + 2:cz + 3] = 3              # enhancing

        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(img, affine), root / "imagesTr" / f"BRATS_{i:03d}.nii.gz")
        nib.save(nib.Nifti1Image(seg, affine), root / "labelsTr" / f"BRATS_{i:03d}.nii.gz")

    (root / "dataset.json").write_text(
        '{"name": "BRAINSynthetic", "modality": {"0": "FLAIR", "1": "T1w", '
        '"2": "T1gd", "3": "T2w"}, '
        '"labels": {"0": "background", "1": "edema", '
        '"2": "non-enhancing tumor", "3": "enhancing tumour"}}'
    )
    return root


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_root):
    from qmmf.manifest import build_manifest
    manifest, report = build_manifest(synthetic_root, progress=False)
    return manifest, report


@pytest.fixture(scope="session")
def synthetic_cache(synthetic_root, tmp_path_factory):
    from qmmf.dataset import CaseCache
    from qmmf.labels import MSD_TASK01
    from qmmf.manifest import discover_cases

    cache = CaseCache(tmp_path_factory.mktemp("cache"), MSD_TASK01)
    cases = discover_cases(synthetic_root)
    for c in cases:
        cache.build(c["case_id"], c["image_path"], c["label_path"])
    return cache, [c["case_id"] for c in cases]
