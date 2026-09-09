"""Deterministic dataset root resolution (plan 5.4).

Fails loudly when the root is ambiguous. Never guesses a mirror silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# Kaggle mount candidates for MSD Task01_BrainTumour, in priority order.
PRIMARY_CANDIDATES: List[str] = [
    "/kaggle/input/medical-segmentation-decathlon-brain-tumour",
    "/kaggle/input/medical-segmentation-decathlon-brain-tumor",
    "/kaggle/working/msd_lite/MSD-BrainTumour",
    "/kaggle/input/msd-task01-braintumour",
]

PEDIATRIC_CANDIDATES: List[str] = [
    "/kaggle/working/brats2023_ped",
    "/kaggle/input/brats2023-ped-dataset",
    "/kaggle/input/brats-peds-2023",
]

HF_PRIMARY = dict(
    repo_id="YongchengYAO/MSD-Lite",
    repo_type="dataset",
    local_dir="/kaggle/working/msd_lite",
    allow_patterns=["MSD-BrainTumour/**", "README.md"],
)

HF_PEDIATRIC = dict(
    repo_id="MedOtter/brats2023-ped-dataset",
    repo_type="dataset",
    local_dir="/kaggle/working/brats2023_ped",
    allow_patterns=["data/nii/BraTS2023_PED/**", "train.jsonl", "README.md"],
)

KAGGLE_SLUG = "thisisrick25/medical-segmentation-decathlon-brain-tumour"


class DatasetRootError(RuntimeError):
    """Raised when the dataset root cannot be resolved unambiguously."""


def _looks_like_msd(root: Path) -> bool:
    return (
        (root / "dataset.json").exists()
        or (root / "imagesTr").is_dir()
        or any(p.name == "imagesTr" for p in root.glob("*/imagesTr"))
    )


def resolve_primary_root(
    candidates: Optional[List[str]] = None, strict: bool = True
) -> Path:
    """Return the MSD Task01 root directory.

    strict=True (the default, and what Notebook 01 uses) raises when zero or
    more than one plausible root exists, rather than picking one.
    """
    candidates = candidates or PRIMARY_CANDIDATES
    found = []
    for c in candidates:
        p = Path(c)
        if not p.exists():
            continue
        if _looks_like_msd(p):
            found.append(p)
        else:
            # One directory level down (Kaggle often nests the archive folder).
            for child in sorted(p.iterdir()):
                if child.is_dir() and _looks_like_msd(child):
                    found.append(child)
    found = sorted(set(found))
    if not found:
        raise DatasetRootError(
            "No MSD Task01_BrainTumour root found. Attach the Kaggle dataset "
            f"'{KAGGLE_SLUG}' or run the Hugging Face snapshot download "
            f"({HF_PRIMARY['repo_id']}). Searched: {candidates}"
        )
    if strict and len(found) > 1:
        raise DatasetRootError(
            f"Ambiguous dataset roots: {[str(p) for p in found]}. "
            "Refusing to guess. Pass an explicit root."
        )
    return found[0]


def resolve_pediatric_root(
    candidates: Optional[List[str]] = None, strict: bool = True
) -> Path:
    candidates = candidates or PEDIATRIC_CANDIDATES
    found = [Path(c) for c in candidates if Path(c).exists()]
    if not found:
        raise DatasetRootError(
            "No BraTS-PEDs 2023 root found. Download the public labelled "
            f"training subset ({HF_PEDIATRIC['repo_id']}) first."
        )
    if strict and len(found) > 1:
        raise DatasetRootError(f"Ambiguous pediatric roots: {found}")
    return found[0]


def download_primary_hf(**kwargs):
    """Hugging Face fallback route; only used when the Kaggle mount is absent."""
    from huggingface_hub import snapshot_download
    args = dict(HF_PRIMARY)
    args.update(kwargs)
    return snapshot_download(**args)


def download_pediatric_hf(**kwargs):
    from huggingface_hub import snapshot_download
    args = dict(HF_PEDIATRIC)
    args.update(kwargs)
    return snapshot_download(**args)
