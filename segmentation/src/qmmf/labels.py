"""Label-map verification and WT/TC/ET derivation.

The plan flags this as a critical hazard (TriMUSE 3.3, QMMF 5.5/6.2): MSD
Task01 label integers are *not* assumed to follow any tutorial convention.
`infer_label_scheme` checks the dataset's own metadata and the data itself, and
`to_nested_targets` refuses to convert under an unverified scheme.

Adult MSD Task01 dataset.json declares:
    0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour
BraTS-style files usually declare:
    0 background, 1 necrotic/non-enhancing core, 2 edema, 4 enhancing tumour
These map to different WT/TC/ET expressions, so the scheme must be resolved
per cohort before any target is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LabelScheme:
    """Which integer label means what, for one cohort."""

    name: str
    edema: int
    non_enhancing: int
    enhancing: int
    background: int = 0

    def component_labels(self) -> Dict[str, int]:
        return {
            "background": self.background,
            "edema": self.edema,
            "non_enhancing": self.non_enhancing,
            "enhancing": self.enhancing,
        }

    def wt_labels(self) -> set:
        return {self.edema, self.non_enhancing, self.enhancing}

    def tc_labels(self) -> set:
        return {self.non_enhancing, self.enhancing}

    def et_labels(self) -> set:
        return {self.enhancing}


# The two schemes seen in practice. Nothing here is applied without verification.
MSD_TASK01 = LabelScheme(name="msd_task01", edema=1, non_enhancing=2, enhancing=3)
BRATS_LEGACY = LabelScheme(name="brats_legacy", edema=2, non_enhancing=1, enhancing=4)

KNOWN_SCHEMES = (MSD_TASK01, BRATS_LEGACY)

# Keyword -> component, used to read dataset.json "labels" dictionaries.
_KEYWORDS = {
    "edema": "edema",
    "oedema": "edema",
    "peritumoral": "edema",
    "non-enhancing": "non_enhancing",
    "nonenhancing": "non_enhancing",
    "necrotic": "non_enhancing",
    "necrosis": "non_enhancing",
    "ncr": "non_enhancing",
    "net": "non_enhancing",
    "core": "non_enhancing",
    "enhancing": "enhancing",
    "background": "background",
}


class LabelSchemeError(RuntimeError):
    """Raised when the label scheme cannot be established from evidence."""


def scheme_from_dataset_json(labels: Mapping) -> LabelScheme:
    """Parse the `labels` block of a dataset.json into a LabelScheme.

    'non-enhancing' is checked before 'enhancing' because the former contains
    the latter as a substring.
    """
    assigned: Dict[str, int] = {}
    for key, value in labels.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            # Some cards invert the mapping: {"edema": "1", ...}
            index = int(value)
            value = key
        text = str(value).strip().lower()
        component = None
        for keyword in ("non-enhancing", "nonenhancing", "necrotic", "necrosis",
                        "ncr", "net", "enhancing", "edema", "oedema",
                        "peritumoral", "background", "core"):
            if keyword in text:
                component = _KEYWORDS[keyword]
                break
        if component is None:
            raise LabelSchemeError(
                f"Unrecognised label description {value!r} for index {index}. "
                "Inspect dataset.json manually before deriving WT/TC/ET."
            )
        if component in assigned and assigned[component] != index:
            raise LabelSchemeError(
                f"Component '{component}' claimed by both label {assigned[component]} "
                f"and {index}."
            )
        assigned[component] = index
    missing = {"edema", "non_enhancing", "enhancing"} - set(assigned)
    if missing:
        raise LabelSchemeError(f"dataset.json is missing components: {sorted(missing)}")
    return LabelScheme(
        name="from_dataset_json",
        background=assigned.get("background", 0),
        edema=assigned["edema"],
        non_enhancing=assigned["non_enhancing"],
        enhancing=assigned["enhancing"],
    )


def verify_scheme_against_data(
    scheme: LabelScheme,
    observed_labels: Iterable[int],
) -> None:
    """Assert that every observed integer belongs to the declared scheme."""
    declared = set(scheme.component_labels().values())
    observed = {int(v) for v in observed_labels}
    unexpected = observed - declared
    if unexpected:
        raise LabelSchemeError(
            f"Labels {sorted(unexpected)} are present in the data but not declared "
            f"by scheme '{scheme.name}' ({sorted(declared)}). "
            "Do not convert until this is resolved."
        )


def infer_label_scheme(
    dataset_json: Optional[Mapping] = None,
    observed_labels: Optional[Iterable[int]] = None,
) -> LabelScheme:
    """Resolve the scheme from metadata when possible, otherwise from observed
    labels only when they are unambiguous. Raises rather than guessing."""
    if dataset_json is not None and "labels" in dataset_json:
        scheme = scheme_from_dataset_json(dataset_json["labels"])
        if observed_labels is not None:
            verify_scheme_against_data(scheme, observed_labels)
        return scheme
    if observed_labels is None:
        raise LabelSchemeError("Provide dataset.json labels or observed label values.")
    observed = {int(v) for v in observed_labels}
    matches = [s for s in KNOWN_SCHEMES if observed <= set(s.component_labels().values())]
    if len(matches) != 1:
        raise LabelSchemeError(
            f"Observed labels {sorted(observed)} match {len(matches)} known schemes. "
            "Manual verification is required (plan 5.5 'Label dictionary')."
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def to_nested_targets(seg: np.ndarray, scheme: LabelScheme) -> np.ndarray:
    """Map an integer segmentation to three overlapping binary channels.

    Returns array of shape (3, ...) ordered (WT, TC, ET), as uint8.
    The nesting ET subset TC subset WT holds by construction.
    """
    seg = np.asarray(seg)
    # Refuse to convert under an unverified scheme: an integer the scheme does
    # not declare means the mirror is not the cohort we think it is.
    verify_scheme_against_data(scheme, np.unique(seg).tolist())
    wt = np.isin(seg, list(scheme.wt_labels()))
    tc = np.isin(seg, list(scheme.tc_labels()))
    et = np.isin(seg, list(scheme.et_labels()))
    stacked = np.stack([wt, tc, et], axis=0).astype(np.uint8)
    assert_nesting(stacked)
    return stacked


def assert_nesting(targets: np.ndarray) -> None:
    """ET subset TC subset WT, enforced on every derived target."""
    wt, tc, et = targets[0].astype(bool), targets[1].astype(bool), targets[2].astype(bool)
    if np.any(et & ~tc):
        raise LabelSchemeError("Nesting violated: ET voxels outside TC.")
    if np.any(tc & ~wt):
        raise LabelSchemeError("Nesting violated: TC voxels outside WT.")


def label_histogram(seg: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(np.asarray(seg), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def region_volumes_ml(
    targets: np.ndarray, spacing_mm: Sequence[float]
) -> Dict[str, float]:
    """Volume of each nested region in millilitres."""
    voxel_ml = float(np.prod(np.asarray(spacing_mm, dtype=np.float64))) / 1000.0
    names = ("wt", "tc", "et")
    return {n: float(targets[i].sum()) * voxel_ml for i, n in enumerate(names)}


def et_present(targets: np.ndarray, min_voxels: int = 1) -> bool:
    """ET presence indicator used for split stratification and empty-region rules."""
    return bool(targets[2].sum() >= min_voxels)
