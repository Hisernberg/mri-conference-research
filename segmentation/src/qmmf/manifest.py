"""Dataset manifest, NIfTI integrity audit and duplicate/overlap detection.

Implements the mandatory audit of plan 5.5 and cells 06-13 of Notebook 01.
This module is deliberately independent of torch so the audit can run on a
CPU-only session.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .labels import (
    LabelScheme, LabelSchemeError, infer_label_scheme, label_histogram,
    region_volumes_ml, to_nested_targets,
)
from .utils import affine_hash, sha256_array, sha256_file, sha256_obj


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def read_manifest(path: str | Path) -> pd.DataFrame:
    """Preserve the 512-bit signature as text instead of numeric CSV inference."""
    return pd.read_csv(path, dtype={"case_id": str, "fingerprint": str})


def load_dataset_json(root: Path) -> Optional[Dict[str, Any]]:
    for candidate in (root / "dataset.json", root / "Task01_BrainTumour" / "dataset.json"):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def discover_cases(root: Path) -> List[Dict[str, str]]:
    """Pair every labelled image with its mask.

    MSD Task01 stores 4-channel images in imagesTr and integer masks in
    labelsTr under the same file name. Files beginning with '.' are macOS
    resource forks in some mirrors and are skipped.
    """
    root = Path(root)
    images_dir = next(
        (p for p in (root / "imagesTr", root / "Task01_BrainTumour" / "imagesTr")
         if p.is_dir()), None,
    )
    labels_dir = next(
        (p for p in (root / "labelsTr", root / "Task01_BrainTumour" / "labelsTr")
         if p.is_dir()), None,
    )
    if images_dir is None or labels_dir is None:
        raise FileNotFoundError(f"imagesTr/labelsTr not found under {root}")

    cases = []
    for img in sorted(images_dir.glob("*.nii*")):
        if img.name.startswith("."):
            continue
        lbl = labels_dir / img.name
        if not lbl.exists():
            warnings.warn(f"No label for {img.name}; excluded from the labelled cohort.")
            continue
        cases.append({
            "case_id": img.name.split(".")[0],
            "image_path": str(img),
            "label_path": str(lbl),
        })
    if not cases:
        raise FileNotFoundError(f"No image/label pairs discovered under {root}")
    return cases


# --------------------------------------------------------------------------- #
# Per-case audit
# --------------------------------------------------------------------------- #
@dataclass
class CaseAudit:
    case_id: str
    image_path: str
    label_path: str
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    record: Dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)


def audit_case(
    case: Dict[str, str],
    scheme: LabelScheme,
    n_modalities: int = 4,
    compute_hashes: bool = True,
    intensity_stats: bool = True,
) -> CaseAudit:
    """Read one case and record geometry, intensity, label and hash evidence."""
    import nibabel as nib

    out = CaseAudit(case["case_id"], case["image_path"], case["label_path"])
    rec: Dict[str, Any] = {
        "case_id": case["case_id"],
        "image_path": case["image_path"],
        "label_path": case["label_path"],
    }
    try:
        img = nib.load(case["image_path"])
        lbl = nib.load(case["label_path"])
    except Exception as exc:  # noqa: BLE001 - a corrupt archive must be reported
        out.fail(f"unreadable: {exc}")
        out.record = rec
        return out

    img_data = np.asanyarray(img.dataobj)
    lbl_data = np.asanyarray(lbl.dataobj)

    rec["image_shape"] = tuple(int(s) for s in img_data.shape)
    rec["label_shape"] = tuple(int(s) for s in lbl_data.shape)
    rec["zooms"] = tuple(float(z) for z in img.header.get_zooms()[:3])
    rec["orientation"] = "".join(nib.aff2axcodes(img.affine))
    rec["label_orientation"] = "".join(nib.aff2axcodes(lbl.affine))
    rec["affine_hash"] = affine_hash(img.affine)
    rec["label_affine_hash"] = affine_hash(lbl.affine)
    rec["dtype"] = str(img_data.dtype)
    rec["file_bytes"] = int(Path(case["image_path"]).stat().st_size)

    # Geometry agreement between image and mask.
    if img_data.ndim != 4:
        out.fail(f"image is {img_data.ndim}D, expected 4D (H,W,D,M)")
    elif img_data.shape[-1] != n_modalities:
        out.fail(f"{img_data.shape[-1]} channels, expected {n_modalities}")
    if img_data.ndim == 4 and img_data.shape[:3] != lbl_data.shape:
        out.fail(f"image {img_data.shape[:3]} vs label {lbl_data.shape} shape mismatch")
    if rec["orientation"] != rec["label_orientation"]:
        out.fail(
            f"orientation mismatch image {rec['orientation']} vs "
            f"label {rec['label_orientation']}"
        )
    if not np.allclose(img.affine, lbl.affine, atol=1e-3):
        out.fail("image and label affines differ beyond tolerance")

    # Intensity sanity per modality.
    if img_data.ndim == 4 and intensity_stats:
        for m in range(img_data.shape[-1]):
            vol = np.asarray(img_data[..., m], dtype=np.float32)
            finite = np.isfinite(vol)
            if not finite.all():
                out.fail(f"modality {m} contains NaN/Inf")
            nz = vol[finite & (vol != 0)]
            rec[f"m{m}_nonzero_frac"] = float(nz.size / max(vol.size, 1))
            if nz.size == 0:
                out.fail(f"modality {m} is all zero")
                continue
            rec[f"m{m}_mean"] = float(nz.mean())
            rec[f"m{m}_std"] = float(nz.std())
            rec[f"m{m}_p05"] = float(np.percentile(nz, 5))
            rec[f"m{m}_p95"] = float(np.percentile(nz, 95))

    # Labels and nested targets.
    hist = label_histogram(lbl_data)
    rec["label_histogram"] = hist
    declared = set(scheme.component_labels().values())
    unexpected = set(hist) - declared
    if unexpected:
        out.fail(f"labels {sorted(unexpected)} outside declared scheme {sorted(declared)}")
    else:
        try:
            targets = to_nested_targets(lbl_data, scheme)
        except LabelSchemeError as exc:
            out.fail(str(exc))
        else:
            vols = region_volumes_ml(targets, rec["zooms"])
            rec["wt_volume_ml"] = vols["wt"]
            rec["tc_volume_ml"] = vols["tc"]
            rec["et_volume_ml"] = vols["et"]
            rec["et_present"] = int(targets[2].sum() > 0)
            rec["et_voxels"] = int(targets[2].sum())
            rec["brain_support_frac"] = float(
                (np.asarray(img_data, dtype=np.float32).sum(axis=-1) != 0).mean()
            ) if img_data.ndim == 4 else float("nan")

    if compute_hashes:
        rec["file_sha256"] = sha256_file(case["image_path"])
        rec["label_file_sha256"] = sha256_file(case["label_path"])
        rec["array_sha256"] = sha256_array(np.asarray(img_data))
        rec["label_array_sha256"] = sha256_array(np.asarray(lbl_data))
        rec["fingerprint"] = perceptual_fingerprint(img_data)

    rec["audit_ok"] = out.ok
    rec["audit_problems"] = "; ".join(out.problems)
    out.record = rec
    return out


def perceptual_fingerprint(volume: np.ndarray, grid: int = 8) -> str:
    """Coarse downsampled signature for *near*-duplicate detection.

    Exact duplicates are caught by array_sha256; this catches re-encodings and
    resamples of the same patient across the adult and pediatric sources.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol.mean(axis=-1)
    nz = vol[vol != 0]
    if nz.size == 0:
        return "empty"
    vol = (vol - nz.mean()) / (nz.std() + 1e-6)
    factors = [max(1, s // grid) for s in vol.shape]
    trimmed = vol[: factors[0] * grid, : factors[1] * grid, : factors[2] * grid]
    small = trimmed.reshape(
        grid, factors[0], grid, factors[1], grid, factors[2]
    ).mean(axis=(1, 3, 5))
    bits = (small > np.median(small)).astype(np.uint8).ravel()
    return "".join(str(int(b)) for b in bits)


# --------------------------------------------------------------------------- #
# Cohort-level manifest
# --------------------------------------------------------------------------- #
def build_manifest(
    root: Path,
    scheme: Optional[LabelScheme] = None,
    limit: Optional[int] = None,
    compute_hashes: bool = True,
    progress: bool = True,
    workers: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audit every labelled case and return (manifest, report).

    The report carries the pass/fail gate that Notebook 02 refuses to train
    without.
    """
    root = Path(root)
    dataset_json = load_dataset_json(root)
    cases = discover_cases(root)
    if limit:
        cases = cases[:limit]

    if scheme is None:
        observed: set = set()
        import nibabel as nib
        for case in cases[: min(20, len(cases))]:
            observed |= set(np.unique(np.asanyarray(nib.load(case["label_path"]).dataobj)).astype(int).tolist())
        scheme = infer_label_scheme(dataset_json, observed)

    from concurrent.futures import ThreadPoolExecutor
    records, problems = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        audits = pool.map(lambda c: audit_case(c, scheme, compute_hashes=compute_hashes), cases)
        for i, audit in enumerate(audits):
            records.append(audit.record)
            if not audit.ok:
                problems.append({"case_id": audit.case_id, "problems": audit.problems})
            if progress and (i + 1) % 25 == 0:
                print(f"  audited {i + 1}/{len(cases)} cases", flush=True)

    manifest = pd.DataFrame.from_records(records)
    dup = duplicate_audit(manifest)
    report = {
        "root": str(root),
        "n_cases": int(len(manifest)),
        "label_scheme": scheme.name,
        "label_scheme_mapping": scheme.component_labels(),
        "dataset_json_present": dataset_json is not None,
        "modality_order_source": (
            dataset_json.get("modality") if dataset_json else "unverified"
        ),
        "n_failed_cases": len(problems),
        "failed_cases": problems,
        "duplicates": dup,
        "geometry": geometry_summary(manifest),
        "manifest_hash": manifest_hash(manifest),
    }
    report["gate_pass"] = bool(
        report["n_failed_cases"] == 0
        and not dup["exact_duplicate_groups"]
        and report["n_cases"] > 0
    )
    return manifest, report


def geometry_summary(manifest: pd.DataFrame) -> Dict[str, Any]:
    if "image_shape" not in manifest:
        return {}
    shapes = manifest["image_shape"].astype(str).value_counts().to_dict()
    zooms = manifest["zooms"].astype(str).value_counts().to_dict()
    orients = manifest.get("orientation", pd.Series(dtype=str)).value_counts().to_dict()
    return {
        "distinct_shapes": shapes,
        "distinct_spacings": zooms,
        "distinct_orientations": orients,
        "spacing_is_uniform": len(zooms) == 1,
    }


def duplicate_audit(
    manifest: pd.DataFrame, fingerprint_threshold: int = 4
) -> Dict[str, Any]:
    """Exact and near-duplicate detection (plan 5.5).

    Near-duplicate pairs are *flagged for manual review*, not auto-removed;
    the Hamming threshold is a screening heuristic, not a proof of identity.
    """
    out: Dict[str, Any] = {"exact_duplicate_groups": [], "near_duplicate_pairs": []}
    if "array_sha256" in manifest:
        groups = manifest.groupby("array_sha256")["case_id"].apply(list)
        out["exact_duplicate_groups"] = [g for g in groups if len(g) > 1]
    if "fingerprint" in manifest:
        fps = manifest[["case_id", "fingerprint"]].dropna()
        codes = {
            r.case_id: np.frombuffer(r.fingerprint.encode(), dtype=np.uint8) - ord("0")
            for r in fps.itertuples() if r.fingerprint != "empty"
        }
        ids = sorted(codes)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if codes[a].shape != codes[b].shape:
                    continue
                dist = int(np.sum(codes[a] != codes[b]))
                if dist <= fingerprint_threshold:
                    out["near_duplicate_pairs"].append(
                        {"a": a, "b": b, "hamming": dist}
                    )
    return out


def cross_source_overlap(
    adult: pd.DataFrame, pediatric: pd.DataFrame, threshold: int = 4
) -> List[Dict[str, Any]]:
    """Screen for the same patient appearing in both cohorts (plan 5.5)."""
    hits: List[Dict[str, Any]] = []
    if "array_sha256" in adult and "array_sha256" in pediatric:
        shared = set(adult["array_sha256"]) & set(pediatric["array_sha256"])
        for h in shared:
            hits.append({
                "kind": "exact",
                "adult": adult.loc[adult.array_sha256 == h, "case_id"].tolist(),
                "pediatric": pediatric.loc[pediatric.array_sha256 == h, "case_id"].tolist(),
            })
    if "fingerprint" in adult and "fingerprint" in pediatric:
        for ra in adult.itertuples():
            fa = getattr(ra, "fingerprint", None)
            if not fa or fa == "empty":
                continue
            va = np.frombuffer(fa.encode(), dtype=np.uint8)
            for rp in pediatric.itertuples():
                fp = getattr(rp, "fingerprint", None)
                if not fp or fp == "empty" or len(fp) != len(fa):
                    continue
                vp = np.frombuffer(fp.encode(), dtype=np.uint8)
                dist = int(np.sum(va != vp))
                if dist <= threshold:
                    hits.append({
                        "kind": "near", "adult": ra.case_id,
                        "pediatric": rp.case_id, "hamming": dist,
                    })
    return hits


def manifest_hash(manifest: pd.DataFrame) -> str:
    cols = [c for c in ("case_id", "array_sha256", "label_array_sha256")
            if c in manifest.columns]
    if not cols:
        cols = ["case_id"]
    payload = manifest[cols].sort_values("case_id").to_dict(orient="records")
    return sha256_obj(payload)[:16]
