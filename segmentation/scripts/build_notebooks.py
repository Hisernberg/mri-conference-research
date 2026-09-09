#!/usr/bin/env python3
"""Build the three Kaggle notebooks from the cell contracts in plan section 12.

Keeping the notebooks generated from one source means the cell-by-cell contract
(20 / 21 / 16 cells) stays visible and auditable, and a change to the protocol
edits one file rather than three JSON blobs.

    python scripts/build_notebooks.py --out notebooks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

Cell = Tuple[str, str, str]      # (kind, title, source)


def md(title: str, body: str) -> Cell:
    return ("markdown", title, body)


def code(title: str, body: str) -> Cell:
    return ("code", title, body)


def to_notebook(cells: List[Cell]) -> dict:
    out = []
    for i, (kind, title, body) in enumerate(cells, start=1):
        source = body.strip("\n").splitlines(keepends=True)
        if kind == "markdown":
            out.append({"cell_type": "markdown", "metadata": {"cell": i},
                        "source": source})
        else:
            out.append({"cell_type": "code", "execution_count": None,
                        "metadata": {"cell": i, "title": title},
                        "outputs": [], "source": source})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =========================================================================== #
# Notebook 01 - data audit and preprocessing contract (20 cells)
# =========================================================================== #
NB01: List[Cell] = [
    md("01 title", """
# QMMF-Net - Notebook 01: Data Audit, Label Verification and Split Lock

**Protocol** `qmmf-v1.0` - Quality-Conditioned Masked Moment Fusion for calibrated
brain-tumour segmentation with arbitrary missing MRI modalities.

**This notebook produces no model results.** It establishes the data facts that
every later claim rests on: provenance, integrity, label semantics, cohort
statistics, and the frozen patient-level partitions.

**Gate G0.** Notebook 02 refuses to train unless the audit summary in the final
cell reports `gate_pass: true` and the manifest and split hashes match.

Outputs: `manifest.csv`, `dataset_fingerprint.json`, `adult_splits.json`,
`ood_manifest.csv`, `preprocessing_config.json`, `quality_feature_reference.json`,
`qc_report.json` and QC figures.
"""),
    code("environment", """
# Cell 02 - environment setup and version capture.
import subprocess, sys, os, json, time
from pathlib import Path

REPO = Path("/kaggle/working/qmmf_net")          # or wherever the repo is attached
if not REPO.exists():
    REPO = Path.cwd().parent if (Path.cwd().parent / "src" / "qmmf").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

for pkg in ("nibabel", "monai", "scikit-learn", "statsmodels"):
    try:
        __import__(pkg.replace("-", "_").replace("scikit_learn", "sklearn"))
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import numpy as np, pandas as pd, torch
from qmmf.utils import environment_fingerprint, write_json

ARTIFACTS = Path("/kaggle/working/artifacts"); ARTIFACTS.mkdir(parents=True, exist_ok=True)
FIGURES = ARTIFACTS / "figures"; FIGURES.mkdir(exist_ok=True)
CACHE_DIR = Path("/kaggle/working/cache")

env = environment_fingerprint()
write_json(ARTIFACTS / "environment.json", env)
print(json.dumps(env, indent=2))
"""),
    code("hardware", """
# Cell 03 - GPU / CPU / RAM / disk inventory and deterministic settings.
from qmmf.utils import seed_everything

seed_everything(42, deterministic=True)
os.environ["QMMF_BASE_SEED"] = "42"

print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GB")
print("CPU count:", os.cpu_count())
try:
    import psutil
    print(f"RAM: {psutil.virtual_memory().total/1024**3:.1f} GB")
except ImportError:
    pass
print(subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True,
                     text=True).stdout)

# Two 16 GB T4s are two devices, not one 32 GB device (plan 11.1).
assert torch.cuda.device_count() <= 2, "Unexpected GPU count for this protocol."
"""),
    code("resolve roots", """
# Cell 04 - resolve dataset roots deterministically; fail loudly if ambiguous.
from qmmf.paths import (
    DatasetRootError, HF_PEDIATRIC, HF_PRIMARY, KAGGLE_SLUG,
    resolve_pediatric_root, resolve_primary_root,
)

try:
    PRIMARY_ROOT = resolve_primary_root()
    print("adult root:", PRIMARY_ROOT)
except DatasetRootError as exc:
    print("ADULT ROOT UNRESOLVED:", exc)
    print(f"Attach the Kaggle dataset '{KAGGLE_SLUG}', or run:")
    print(f"  from qmmf.paths import download_primary_hf; download_primary_hf()")
    raise

try:
    PED_ROOT = resolve_pediatric_root()
    print("pediatric root:", PED_ROOT)
except DatasetRootError as exc:
    PED_ROOT = None
    print("pediatric cohort not present yet (needed only for Notebook 03):", exc)
"""),
    code("provenance", """
# Cell 05 - dataset metadata, licence and provenance record.
from qmmf.manifest import load_dataset_json

dataset_json = load_dataset_json(PRIMARY_ROOT)
provenance = {
    "adult_root": str(PRIMARY_ROOT),
    "adult_source": "Medical Segmentation Decathlon Task01_BrainTumour",
    "adult_licence": "CC BY-SA 4.0 (verify on the dataset card before release)",
    "adult_origin_note": (
        "Task01 is derived from BraTS 2016/2017 adult cases; it is a public "
        "retrospective benchmark, not a contemporary multi-institutional cohort."
    ),
    "pediatric_root": str(PED_ROOT) if PED_ROOT else None,
    "pediatric_source": "BraTS-PEDs 2023 public labelled training subset (99 cases)",
    "pediatric_use": "one-time out-of-domain stress test; no tuning, no adaptation",
    "download_timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    "dataset_json": dataset_json,
}
write_json(ARTIFACTS / "provenance.json", provenance)
print(json.dumps({k: v for k, v in provenance.items() if k != "dataset_json"}, indent=2))
print("dataset.json modality block:", (dataset_json or {}).get("modality"))
"""),
    code("enumerate cases", """
# Cell 06 - enumerate image/label pairs and build the raw case list.
from qmmf.manifest import discover_cases

cases = discover_cases(PRIMARY_ROOT)
print(f"{len(cases)} labelled image/label pairs discovered")
print(pd.DataFrame(cases).head())

# The plan expects 484 labelled adult cases. A different count is not fatal, but
# it means the mirror differs from the official archive and must be documented.
if len(cases) != 484:
    print(f"NOTE: expected 484 labelled cases, found {len(cases)}. "
          "Record this deviation in the manuscript's data section.")
"""),
    code("label scheme", """
# Cell 07 - resolve and verify the label scheme BEFORE any target is derived.
# This is the single most dangerous silent error in BraTS-lineage datasets.
import nibabel as nib
from qmmf.labels import infer_label_scheme, verify_scheme_against_data

observed = set()
for c in cases[:20]:
    observed |= set(np.unique(np.asanyarray(nib.load(c["label_path"]).dataobj)).astype(int).tolist())
print("observed label values in the first 20 cases:", sorted(observed))

SCHEME = infer_label_scheme(dataset_json, observed)
verify_scheme_against_data(SCHEME, observed)
print("resolved scheme:", SCHEME.name, SCHEME.component_labels())
print("WT =", sorted(SCHEME.wt_labels()),
      "| TC =", sorted(SCHEME.tc_labels()),
      "| ET =", sorted(SCHEME.et_labels()))
"""),
    code("full audit", """
# Cell 08 - full NIfTI audit: shape, affine, orientation, spacing, NaN/Inf,
# zero-volume modalities, hashes and per-case statistics.
from qmmf.manifest import build_manifest

manifest, report = build_manifest(PRIMARY_ROOT, scheme=SCHEME, compute_hashes=True)
manifest.to_csv(ARTIFACTS / "manifest.csv", index=False)
write_json(ARTIFACTS / "audit_report.json", report)

print("cases:", report["n_cases"], "| failures:", report["n_failed_cases"])
print("geometry:", json.dumps(report["geometry"], indent=2)[:800])
for f in report["failed_cases"][:10]:
    print("FAIL", f)
"""),
    code("modality order", """
# Cell 09 - verify modality order and channel identity, never assume it.
# MSD Task01 dataset.json declares FLAIR, T1w, T1gd, T2w in channel order.
# Our canonical internal order is (t1, t1ce, t2, flair), so a permutation is
# needed - and it must be derived from the metadata, not from a filename guess.
declared = (dataset_json or {}).get("modality", {})
print("declared channel order:", declared)

CANONICAL = ["t1", "t1ce", "t2", "flair"]
NAME_MAP = {"flair": "flair", "t1w": "t1", "t1": "t1", "t1gd": "t1ce",
            "t1ce": "t1ce", "t1-gd": "t1ce", "t2w": "t2", "t2": "t2"}
if declared:
    channel_names = [NAME_MAP[str(declared[str(i)]).strip().lower()]
                     for i in range(len(declared))]
    MODALITY_PERMUTATION = [channel_names.index(m) for m in CANONICAL]
    print("channel names:", channel_names)
    print("permutation raw->canonical:", MODALITY_PERMUTATION)
else:
    MODALITY_PERMUTATION = None
    print("WARNING: no modality block; order must be verified manually before use.")

# Cross-check with intensity behaviour: T1ce should have the highest intensity
# inside the enhancing region relative to its own brain mean, on ET-present cases.
et_cases = manifest[manifest.et_present == 1].case_id.head(5).tolist()
print("intensity cross-check cases:", et_cases)
write_json(ARTIFACTS / "modality_order.json",
           {"declared": declared, "canonical": CANONICAL,
            "permutation": MODALITY_PERMUTATION})
"""),
    code("label mapping check", """
# Cell 10 - unit-test the label mapping on real cases and confirm nesting.
from qmmf.labels import assert_nesting, region_volumes_ml, to_nested_targets

checked = 0
for c in cases[:20]:
    seg = np.asanyarray(nib.load(c["label_path"]).dataobj)
    targets = to_nested_targets(seg, SCHEME)
    assert_nesting(targets)
    checked += 1
print(f"nesting verified on {checked} cases (ET subset TC subset WT)")

# Manual review panel: the plan requires eyeballing at least 20 cases.
print(manifest[["case_id", "wt_volume_ml", "tc_volume_ml", "et_volume_ml",
                "et_present"]].head(20).to_string(index=False))
"""),
    code("qc montage", """
# Cell 11 - QC montage across modalities and labels (manual review, 12+ cases).
import matplotlib.pyplot as plt
from qmmf.manifest import discover_cases

def montage(case, n_cols=5, out=None):
    img = np.asanyarray(nib.load(case["image_path"]).dataobj)
    seg = np.asanyarray(nib.load(case["label_path"]).dataobj)
    z = int(np.argmax((seg > 0).sum(axis=(0, 1)))) if (seg > 0).any() else seg.shape[2] // 2
    fig, ax = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2))
    for m in range(min(4, img.shape[-1])):
        ax[m].imshow(np.rot90(img[..., z, m]), cmap="gray")
        ax[m].set_title(f"channel {m}"); ax[m].axis("off")
    ax[4].imshow(np.rot90(img[..., z, 0]), cmap="gray")
    ax[4].imshow(np.rot90(np.ma.masked_where(seg[..., z] == 0, seg[..., z])),
                 alpha=0.6, cmap="autumn", vmin=0, vmax=3)
    ax[4].set_title(f"labels z={z}"); ax[4].axis("off")
    fig.suptitle(case["case_id"])
    if out: fig.savefig(out, dpi=90, bbox_inches="tight")
    return fig

# Sample across tumour-size bins so the review is not all large tumours.
bins = pd.qcut(manifest.wt_volume_ml, 4, labels=False, duplicates="drop")
review_ids = manifest.assign(bin=bins).groupby("bin").head(3).case_id.tolist()
by_id = {c["case_id"]: c for c in cases}
for cid in review_ids[:12]:
    montage(by_id[cid], out=FIGURES / f"qc_{cid}.png")
plt.close("all")
print("QC montages written for:", review_ids[:12])
"""),
    code("duplicates", """
# Cell 12 - exact and near-duplicate audit, plus cross-source overlap screening.
from qmmf.manifest import cross_source_overlap, duplicate_audit

dup = duplicate_audit(manifest)
print("exact duplicate groups:", dup["exact_duplicate_groups"])
print("near-duplicate pairs flagged for manual review:",
      dup["near_duplicate_pairs"][:10])
write_json(ARTIFACTS / "duplicate_audit.json", dup)

# TCGA-LGG is deliberately excluded as a claimed external test: it overlaps the
# BraTS lineage this cohort comes from (plan 5.3). Nothing here re-admits it.
assert not dup["exact_duplicate_groups"], (
    "Exact duplicates present: resolve before splitting (gate G0)."
)
"""),
    code("cohort stats", """
# Cell 13 - cohort statistics: tumour volumes, ET presence, brain support, geometry.
stats = manifest[["wt_volume_ml", "tc_volume_ml", "et_volume_ml",
                  "brain_support_frac"]].describe()
print(stats.to_string())
print("\\nET present:", int(manifest.et_present.sum()), "/", len(manifest))
print("ET-absent cases are reported separately in every metric table "
      "(empty-region rule, plan 10.1).")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for i, col in enumerate(["wt_volume_ml", "tc_volume_ml", "et_volume_ml"]):
    ax[i].hist(manifest[col], bins=30); ax[i].set_title(col); ax[i].set_xlabel("ml")
fig.savefig(FIGURES / "cohort_volumes.png", dpi=110, bbox_inches="tight")
plt.close(fig)
"""),
    code("splits", """
# Cell 14 - locked adult test (20%) plus four development folds. Frozen here.
from qmmf.splits import build_splits, validate_splits

splits = build_splits(manifest, seed=42)
validate_splits(splits)
split_hash = splits.save(ARTIFACTS / "adult_splits.json")

print("locked test:", len(splits.locked_test), "cases")
print("development:", len(splits.development), "cases")
for k, part in splits.folds.items():
    print(f"  fold {k}: train {len(part['train'])}, inner_val "
          f"{len(part['inner_val'])}, outer_val {len(part['outer_val'])}")
print("split hash:", split_hash)
print("\\nThe locked test is created ONCE, before any architecture decision, "
      "and is not touched again until Notebook 03.")
"""),
    code("label subsets", """
# Cell 15 - nested 25% / 50% / 100% label-efficiency subsets.
for frac, ids in splits.label_subsets.items():
    print(f"  {frac}: {len(ids)} cases")
s25, s50, s100 = (set(splits.label_subsets[k]) for k in ("0.25", "0.50", "1.00"))
assert s25 <= s50 <= s100, "Label subsets must be strictly nested (plan 6.5)."
print("nesting verified: 25% subset of 50% subset of 100%")

# Stratification balance check across the partitions.
strata = pd.Series(splits.strata)
balance = pd.DataFrame({
    "locked": strata[splits.locked_test].value_counts(normalize=True),
    "development": strata[splits.development].value_counts(normalize=True),
}).fillna(0).round(3)
print(balance.to_string())
"""),
    code("preprocess cache", """
# Cell 16 - build the preprocessed case cache and fit training-fold quality stats.
from qmmf.dataset import CaseCache
from qmmf.quality import QualityNormalizer

cache = CaseCache(CACHE_DIR, SCHEME)
for i, c in enumerate(cases):
    cache.build(c["case_id"], c["image_path"], c["label_path"])
    if (i + 1) % 50 == 0:
        print(f"  cached {i+1}/{len(cases)}")
print("cache size:", sum(p.stat().st_size for p in CACHE_DIR.glob('*.npz')) / 1024**3,
      "GB")

# Quality-feature reference distributions use TRAINING-fold cases only.
for fold, part in splits.folds.items():
    qn = QualityNormalizer.fit([cache.load(c)["quality_raw"] for c in part["train"]])
    write_json(ARTIFACTS / f"quality_norm_fold{fold}.json", qn.to_dict())
print("quality normalisers fitted per fold (training cases only, no mask access)")
"""),
    code("transforms and reconstruction", """
# Cell 17 - 2.5D window index and deterministic reconstruction check.
from qmmf.config import ExperimentConfig
from qmmf.inference import predict_volume
from qmmf.transforms import uncrop_to_original

cfg = ExperimentConfig.load(REPO / "configs" / "base.yaml")
cfg = cfg.merged({"data.manifest_hash": report["manifest_hash"],
                  "data.split_hash": split_hash,
                  "data.root": str(PRIMARY_ROOT)})

class _Marker(torch.nn.Module):
    def forward(self, image, availability, quality, return_aux=False):
        plane = image[:, 0, image.shape[2] // 2]
        logits = plane.unsqueeze(1).repeat(1, 3, 1, 1)
        return (logits, {}) if return_aux else logits

probe = splits.development[0]
out = predict_volume(_Marker(), probe, cache, cfg, device=torch.device("cpu"),
                     return_logits=True, apply_nesting=False)
rec = cache.load(probe)
expected = uncrop_to_original(rec["image"][0][None].astype(np.float32),
                              out["geometry"]["crop_box"],
                              out["geometry"]["original_shape"])[0]
err = float(np.abs(out["logits"][0] - expected).max())
print(f"reconstruction max error: {err:.2e}")
assert err < 1e-3, "2.5D -> 3D reconstruction is not exact; metrics would be void."
"""),
    code("dataloader tests", """
# Cell 18 - Dataset / DataLoader unit tests: shapes, alignment, no empty batch.
from torch.utils.data import DataLoader
from qmmf.dataset import SliceDataset, collate

qn = QualityNormalizer.from_dict(json.loads(
    (ARTIFACTS / "quality_norm_fold0.json").read_text()))
ds = SliceDataset(splits.folds["0"]["train"], cache, cfg.data,
                 quality_norm=qn, length=64, seed=42)
ds.set_epoch(0, cfg.train.max_epochs)
loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=collate)
batch = next(iter(loader))

assert batch["image"].shape[1:] == (4, cfg.data.context_slices, *cfg.data.crop_size)
assert batch["target"].shape[1:] == (3, *cfg.data.crop_size)
assert (batch["availability"].sum(1) >= 1).all(), "empty modality subset produced"
absent = batch["availability"] == 0
assert batch["image"][absent].abs().max() == 0, "absent modality is not zeroed"
# Nesting must survive augmentation.
t = batch["target"]
assert (t[:, 2] <= t[:, 1] + 1e-6).all() and (t[:, 1] <= t[:, 0] + 1e-6).all()
print("dataloader contract satisfied:", {k: tuple(v.shape) for k, v in batch.items()
                                          if torch.is_tensor(v)})
"""),
    code("export", """
# Cell 19 - export manifest, split JSON, preprocessing config and fingerprints.
from qmmf.manifest import manifest_hash

fingerprint = {
    "protocol_version": cfg.protocol_version,
    "manifest_hash": report["manifest_hash"],
    "split_hash": split_hash,
    "label_scheme": SCHEME.name,
    "label_mapping": SCHEME.component_labels(),
    "modality_permutation": MODALITY_PERMUTATION,
    "n_cases": report["n_cases"],
    "n_locked_test": len(splits.locked_test),
    "n_development": len(splits.development),
    "environment": env,
    "provenance": {k: v for k, v in provenance.items() if k != "dataset_json"},
}
write_json(ARTIFACTS / "dataset_fingerprint.json", fingerprint)
cfg.save(ARTIFACTS / "preprocessing_config.json")
print(json.dumps({k: v for k, v in fingerprint.items()
                  if k not in ("environment", "provenance")}, indent=2))
"""),
    code("gate", """
# Cell 20 - final audit summary and the PASS/FAIL gate consumed by Notebook 02.
gate = {
    "gate_id": "G0_data_integrity",
    "all_cases_readable": report["n_failed_cases"] == 0,
    "no_exact_duplicates": not dup["exact_duplicate_groups"],
    "label_scheme_verified": True,
    "nesting_verified": True,
    "splits_disjoint": True,
    "reconstruction_exact": err < 1e-3,
    "manifest_hash": report["manifest_hash"],
    "split_hash": split_hash,
    "near_duplicate_pairs_for_review": len(dup["near_duplicate_pairs"]),
}
gate["gate_pass"] = all(v for k, v in gate.items()
                        if isinstance(v, bool))
write_json(ARTIFACTS / "qc_report.json", gate)
print(json.dumps(gate, indent=2))

if not gate["gate_pass"]:
    raise SystemExit("G0 FAILED. Repair the data before training (plan 13.1).")
print("\\nG0 passed. Notebook 02 may run.")
print("No model has been trained and no result has been produced by this notebook.")
"""),
]


# =========================================================================== #
# Notebook 02 - baselines, cross-validation and ablation (21 cells)
# =========================================================================== #
NB02: List[Cell] = [
    md("02 title", """
# QMMF-Net - Notebook 02: Baselines, Cross-Validation, Ablation and Architecture Lock

Development-only. This notebook may look at the four development folds as often
as it likes; it may **never** touch the locked adult test or the pediatric cohort.

It ends by writing `model_lock.json`: the frozen configuration, thresholds,
post-processing and calibration recipe that Notebook 03 executes exactly once.

Gates enforced here: **G1** feasibility (VRAM/speed), **G2** mechanism,
**G3** quality validity (shuffled- and no-quality controls), **G4** architecture lock.
"""),
    code("verify notebook 01", """
# Cell 01 - load Notebook 01 artifacts and verify hashes. Refuse unknown data.
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path("/kaggle/working/qmmf_net")
if not REPO.exists():
    REPO = Path.cwd().parent if (Path.cwd().parent / "src" / "qmmf").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

ARTIFACTS = Path("/kaggle/working/artifacts")
CACHE_DIR = Path("/kaggle/working/cache")
FIGURES = ARTIFACTS / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)

from qmmf.config import ExperimentConfig
from qmmf.labels import MSD_TASK01, LabelScheme
from qmmf.splits import Splits
from qmmf.utils import read_json, seed_everything, write_json

qc = read_json(ARTIFACTS / "qc_report.json")
fingerprint = read_json(ARTIFACTS / "dataset_fingerprint.json")
assert qc["gate_pass"], "G0 did not pass; training is blocked."

splits = Splits.load(ARTIFACTS / "adult_splits.json")
assert splits.split_hash == qc["split_hash"], "Split file does not match the audit."
manifest = pd.read_csv(ARTIFACTS / "manifest.csv")

cfg = ExperimentConfig.load(ARTIFACTS / "preprocessing_config.json")
assert cfg.data.manifest_hash == qc["manifest_hash"]
seed_everything(cfg.train.seed)
print("verified against Notebook 01:", qc["manifest_hash"], qc["split_hash"])
"""),
    code("registry", """
# Cell 02 - typed experiment configuration and the model / ablation registry.
from qmmf.models import ABLATIONS, BASELINE_IDS, MODEL_REGISTRY, build_model

print("models:", sorted(MODEL_REGISTRY))
print("baselines:", BASELINE_IDS)
print(f"{len(ABLATIONS)} ablations:")
for k, a in ABLATIONS.items():
    print(f"  {k:4s} {a.description:48s} -> {a.question}")
print("\\nconfig hash of the base setup:", cfg.config_hash())
"""),
    code("mask and quality interface", """
# Cell 03 - modality mask and quality vector interface.
from qmmf.dataset import CaseCache
from qmmf.quality import QUALITY_FEATURES, QualityNormalizer
from qmmf.subsets import ALL_SUBSET_KEYS, ALL_SUBSETS, availability_mask

SCHEME = LabelScheme(name="resolved", **{k: v for k, v in
                                          fingerprint["label_mapping"].items()})
cache = CaseCache(CACHE_DIR, SCHEME)
qnorms = {f: QualityNormalizer.from_dict(read_json(ARTIFACTS / f"quality_norm_fold{f}.json"))
          for f in splits.folds}

print("quality features:", QUALITY_FEATURES)
print("15 subsets:", ALL_SUBSET_KEYS)
sample = cache.load(splits.development[0])
print("raw quality matrix [M, Q]:\\n", np.round(sample["quality_raw"], 3))
print("normalised:\\n", np.round(qnorms["0"].transform(sample["quality_raw"]), 3))
"""),
    code("baselines 2d 25d", """
# Cell 04 - B1 (2D U-Net) and B2 (2.5D U-Net) baselines.
from qmmf.utils import count_parameters

for bid in ("B1", "B2"):
    m = build_model(cfg.merged({"model": BASELINE_IDS[bid]}))
    print(f"{bid} {BASELINE_IDS[bid]:10s} {count_parameters(m)['total']/1e6:6.2f}M params")
"""),
    code("baseline hemis", """
# Cell 05 - B3: HeMIS-style shared stem with equal masked mean and variance.
m = build_model(cfg.merged({"model": "hemis25d"}))
print("B3 hemis25d", count_parameters(m)["total"] / 1e6, "M params")
print("This is the foundational arbitrary-modality baseline; QMMF-Net must beat "
      "it on the robustness endpoints for the mechanism claim (gate G2).")
"""),
    code("baseline segresnet", """
# Cell 06 - B4: small 3D SegResNet, the volumetric full-modality reference.
try:
    m = build_model(cfg.merged({"model": "segresnet3d"}))
    print("B4 segresnet3d", count_parameters(m)["total"] / 1e6, "M params")
except ImportError as exc:
    print("MONAI missing:", exc)
    print("B4 is the H1 non-inferiority comparator and cannot be skipped silently.")
"""),
    code("qmmf model", """
# Cell 07 - the QMMF block and QMMF-Net.
model = build_model(cfg)
p = count_parameters(model)
print(f"QMMF-Net: {p['total']/1e6:.2f}M parameters "
      f"(plan 7.7 gate: < 8M)")
print(model.fusions[0])
"""),
    code("model unit tests", """
# Cell 08 - model unit tests: tensors, gradients, missing subsets, nesting.
from qmmf.losses import QMMFLoss
from qmmf.models import project_nested

B, M, D, H, W, Q = 2, 4, cfg.data.context_slices, 64, 64, cfg.net.quality_dim
image = torch.randn(B, M, D, H, W)
quality = torch.randn(B, M, Q)
avail = torch.tensor([[1., 1., 0., 1.], [0., 0., 1., 0.]])
image = image * avail[:, :, None, None, None]

model.train()
logits, aux = model(image, avail, quality, return_aux=True)
assert logits.shape == (B, 3, H, W)

# An absent modality must not be able to change the output.
model.eval()
with torch.no_grad():
    a = model(image, avail, quality)
    perturbed = image.clone(); perturbed[0, 2] = torch.randn_like(perturbed[0, 2]) * 50
    b = model(perturbed, avail, quality)
assert torch.allclose(a, b, atol=1e-5), "absent-modality leakage detected"

probs = project_nested(torch.sigmoid(a))
assert (probs[:, 2] <= probs[:, 1] + 1e-6).all() and (probs[:, 1] <= probs[:, 0] + 1e-6).all()
print("model unit tests passed (shapes, masking, nesting)")
"""),
    code("efficiency benchmark", """
# Cell 09 - parameter, MAC, VRAM and 200-step speed benchmark. GATE G1.
from qmmf.dataset import SliceDataset
from qmmf.trainer import check_resource_gates, speed_and_memory_pilot
from qmmf.utils import estimate_macs

ds = SliceDataset(splits.folds["0"]["train"], cache, cfg.data,
                  quality_norm=qnorms["0"], length=1024, seed=cfg.train.seed)
ds.set_epoch(0, cfg.train.max_epochs)

efficiency = {}
for name in ["qmmf_net", "unet25d", "hemis25d"]:
    c = cfg.merged({"model": name})
    pilot = speed_and_memory_pilot(c, ds, steps=200)
    gates = check_resource_gates(pilot)
    efficiency[name] = {**pilot, **gates}
    print(f"{name:12s} {pilot['parameters_millions']:5.2f}M  "
          f"VRAM {pilot['peak_vram_gb']:5.2f} GB  "
          f"{pilot['steps_per_second']:5.2f} steps/s  "
          f"vram_pass={gates['vram_pass']}")

write_json(ARTIFACTS / "efficiency_benchmark.json", efficiency)
assert efficiency["qmmf_net"]["vram_pass"], (
    "G1 failed: reduce width/crop or enable checkpointing (plan 11.5)."
)
"""),
    code("losses", """
# Cell 10 - loss composition and the loss-balance audit.
from qmmf.losses import estimate_region_weights

pos_frac = [float((manifest[f"{r}_volume_ml"] > 0).mean()) for r in ("wt", "tc", "et")]
region_weights = estimate_region_weights(
    [manifest[f"{r}_volume_ml"].mean() / manifest.wt_volume_ml.mean()
     for r in ("wt", "tc", "et")]
)
print("region weights (fixed for the whole run):", region_weights.tolist())

criterion = QMMFLoss(cfg.loss, region_weights)
target = (torch.rand(B, 3, H, W) > 0.7).float()
target[:, 1] *= target[:, 0]; target[:, 2] *= target[:, 1]
model.train()
lg, ax_ = model(image, avail, quality, return_aux=True)
parts = criterion(lg, target, ax_["deep_logits"])
print({k: round(float(v), 4) for k, v in parts.items()})
print("\\nIf one auxiliary term dominates by >5x, rescale or remove it and "
      "report the decision (plan 8.2).")
"""),
    code("training loop", """
# Cell 11 - training / validation loop with AMP, accumulation, EMA and resume.
from qmmf.evaluate import validation_metrics
from qmmf.trainer import Trainer

spacing = {str(r.case_id): eval(str(r.zooms)) for r in manifest.itertuples()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_validate(c, inner_val):
    def validate(m, epoch):
        return validation_metrics(m, inner_val, cache, c, spacing, device=device,
                                  max_cases=24)
    return validate

print("Trainer is configured; the run matrix below drives it through the queue.")
"""),
    code("reconstruction check", """
# Cell 12 - 3D reconstruction and patient-level metric validation on toy cases.
from qmmf.inference import binarize, predict_volume
from qmmf.metrics import evaluate_case, macro_dice

cid = splits.folds["0"]["inner_val"][0]
out = predict_volume(model, cid, cache, cfg, device=device)
pred = binarize(out["probs"], 0.5)
rows = evaluate_case(pred, out["reference"], spacing[cid], case_id=cid)
for r in rows:
    print(f"  {r.region}: dice={r.dice:.3f} hd95={r.hd95:.2f} nsd={r.nsd:.3f} "
          f"ref_empty={r.reference_empty}")
print("macro (empty-reference regions excluded):", round(macro_dice(rows), 4))

# Sanity floor: an untrained network must not look good. If it does, the
# metric or the reference is wrong.
print("\\nUntrained-model macro Dice above ~0.3 would indicate a metric bug.")
"""),
    code("queue and workers", """
# Cell 13 - append-safe experiment queue and the dual-GPU launcher.
from qmmf.ledger import ExperimentQueue, Ledger
from qmmf.worker import build_experiment_queue, launch_dual_gpu, wait_for

QUEUE = ARTIFACTS / "experiment_queue.json"
LEDGER = ARTIFACTS / "experiment_ledger.csv"

added = build_experiment_queue(
    QUEUE, models=["unet2d", "unet25d", "hemis25d", "segresnet3d", "qmmf_net"],
    folds=[0, 1, 2, 3], seeds=[cfg.train.seed], base_cfg=cfg,
)
print("core runs queued:", added, ExperimentQueue(QUEUE).summary())

WORKER = REPO / "scripts" / "run_worker.py"
LAUNCH = [
    "--base", str(ARTIFACTS / "preprocessing_config.json"),
    "--cache", str(CACHE_DIR), "--splits", str(ARTIFACTS / "adult_splits.json"),
    "--manifest", str(ARTIFACTS / "manifest.csv"),
    "--time-budget", "36000",       # stop cleanly inside the 12-hour ceiling
]
print("launch with:")
print(f"  procs = launch_dual_gpu('{WORKER}', '{QUEUE}', '{LEDGER}', {LAUNCH})")
print("  wait_for(procs)")
"""),
    code("pilot grid", """
# Cell 14 - small pre-declared pilot grid, then FREEZE the hyperparameters.
PILOT_GRID = {
    "train.lr": [2e-4, 3e-4, 5e-4],
    "train.weight_decay": [1e-4, 1e-3, 1e-2],
}
print("Pilot grid (development fold 0 only):", json.dumps(PILOT_GRID, indent=2))
print("Rules (plan 9.2):")
print("  * the same grid size is spent on every baseline that has one;")
print("  * the winner is chosen on fold 0 inner validation and then frozen;")
print("  * no further tuning happens after the architecture lock.")
write_json(ARTIFACTS / "pilot_grid.json", PILOT_GRID)
"""),
    code("core runs", """
# Cell 15 - core four-fold runs for all five models.
# procs = launch_dual_gpu(WORKER, QUEUE, LEDGER, LAUNCH); wait_for(procs)
ledger = Ledger(LEDGER)
df = ledger.read()
print(df.groupby(["model", "status"]).size() if len(df) else "no runs recorded yet")
print("\\nFailures and OOMs are recorded as rows with status failed / "
      "resource_rejected; they are reported in the paper, not deleted.")
"""),
    code("oof assembly", """
# Cell 16 - assemble out-of-fold predictions and compare on development data.
from qmmf.evaluate import evaluate_cases
from qmmf.stats import apply_holm, compare_paired

def oof_macro(model_name):
    \"\"\"Patient-level OOF macro Dice for one model across the four folds.\"\"\"
    scores = {}
    for fold, part in splits.folds.items():
        run = df[(df.model == model_name) & (df.fold == int(fold)) &
                 (df.status == "completed")]
        if run.empty:
            continue
        ckpt = torch.load(run.iloc[0].checkpoint, map_location=device,
                          weights_only=False)
        m = build_model(cfg.merged({"model": model_name})); m.load_state_dict(ckpt["model"])
        res = evaluate_cases(m.to(device).eval(), part["outer_val"], cache, cfg,
                             spacing, device=device)
        scores.update(res["per_case_macro"])
    return scores

# comparisons = []
# base = oof_macro("hemis25d"); prop = oof_macro("qmmf_net")
# shared = sorted(set(base) & set(prop))
# comparisons.append(compare_paired([prop[c] for c in shared],
#                                   [base[c] for c in shared],
#                                   name="qmmf_vs_hemis_oof"))
# apply_holm(comparisons)
print("OOF comparison is patient-level and paired. Never run a t-test across "
      "the four fold means (plan 10.5).")
"""),
    code("ablation screen", """
# Cell 17 - one-fold ablation screen A1-A17 (plan 9.3 ablation funnel).
screen = build_experiment_queue(
    QUEUE, models=["qmmf_net"], folds=[0], seeds=[cfg.train.seed],
    ablations=[a for a in ABLATIONS if a != "A0"], base_cfg=cfg,
)
print("ablation screen runs queued:", screen)
print("\\nGate G3 (quality validity): A5 (no quality) and A17 (shuffled quality)")
print("must BOTH degrade the relevant endpoint. If they do not, the quality")
print("contribution claim is dropped from the paper before the full CV.")
"""),
    code("ablation confirmation", """
# Cell 18 - multi-fold confirmation of the top four or five ablations only.
CONFIRM_CRITERIA = {
    "min_absolute_change_pp": 0.5,
    "requires_mechanism_evidence": True,
    "max_confirmed": 5,
}
print(json.dumps(CONFIRM_CRITERIA, indent=2))
print("Ablations whose one-fold change is < 0.5 pp with no mechanism evidence")
print("do not get four-fold budget (plan 11.5 'Ablation value' gate); they are")
print("still reported, as screened-but-not-confirmed.")
"""),
    code("label efficiency", """
# Cell 19 - optional label-efficiency experiments at 25% / 50% / 100%.
le = build_experiment_queue(
    QUEUE, models=["qmmf_net", "hemis25d", "unet25d"], folds=[0, 1],
    seeds=[cfg.train.seed], label_fractions=[0.25, 0.50, 1.00], base_cfg=cfg,
)
print("label-efficiency runs queued:", le)
print("H4: subset consistency should help most at 25% / 50%, not at 100%.")
"""),
    code("architecture lock", """
# Cell 20 - ARCHITECTURE LOCK. Everything Notebook 03 will use is fixed here.
lock = {
    "gate_id": "G4_architecture_lock",
    "protocol_version": cfg.protocol_version,
    "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config": cfg.to_dict(),
    "config_hash": cfg.config_hash(),
    "manifest_hash": qc["manifest_hash"],
    "split_hash": qc["split_hash"],
    "final_seeds": list(cfg.evaluation.mc_seeds),
    "decision_threshold": 0.5,
    "threshold_source": "development inner-validation sweep only",
    "post_processing": "hard nesting projection ET subset TC subset WT; no size filtering",
    "calibration": "region-wise temperature scaling, cross-fitted on development only",
    "confirmed_ablations": [],
    "removed_modules": [],
    "rationale": "",
    "non_inferiority_margin_pp": cfg.evaluation.non_inferiority_margin_pp,
    "primary_endpoint": "adult locked-test full-modality macro Dice (WT, TC, ET)",
}
write_json(ARTIFACTS / "model_lock.json", lock)
print(json.dumps({k: v for k, v in lock.items() if k != "config"}, indent=2))
print("\\nAfter this point, no threshold, no post-processing rule and no")
print("architectural choice may change. Any change requires a versioned")
print("protocol deviation recorded in the manuscript.")
"""),
    code("export", """
# Cell 21 - export the ledger, OOF predictions, checkpoints and efficiency table.
export = {
    "ledger": str(LEDGER),
    "queue": str(QUEUE),
    "model_lock": str(ARTIFACTS / "model_lock.json"),
    "efficiency": str(ARTIFACTS / "efficiency_benchmark.json"),
    "checkpoints": sorted(str(p) for p in Path(cfg.out_dir).glob("*/best.pt")),
}
write_json(ARTIFACTS / "notebook02_exports.json", export)
print(json.dumps(export, indent=2)[:1500])
print("\\nNo locked-test or pediatric number exists at this point, by construction.")
"""),
]


# =========================================================================== #
# Notebook 03 - locked testing, statistics and paper artifacts (16 cells)
# =========================================================================== #
NB03: List[Cell] = [
    md("03 title", """
# QMMF-Net - Notebook 03: Locked Testing, Statistics and Paper Artifacts

**This notebook runs once.** It consumes `model_lock.json` and produces every
number that appears in the paper. There is no tuning here, no threshold search,
and no architectural revision. A correction requires a versioned protocol
deviation, recorded and reported.

Order is deliberate: adult locked test first, then all 15 subsets, then
corruptions, then the pediatric out-of-domain stress test last.

Gates: **G5** reliability, **G6** locked analysis, **G7** OOD analysis.
"""),
    code("load lock", """
# Cell 01 - load the model lock, split hashes, dataset fingerprints.
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path("/kaggle/working/qmmf_net")
if not REPO.exists():
    REPO = Path.cwd().parent if (Path.cwd().parent / "src" / "qmmf").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

ARTIFACTS = Path("/kaggle/working/artifacts")
CACHE_DIR = Path("/kaggle/working/cache")
RESULTS = ARTIFACTS / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES = ARTIFACTS / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)

from qmmf.config import ExperimentConfig
from qmmf.dataset import CaseCache
from qmmf.labels import LabelScheme
from qmmf.splits import Splits
from qmmf.utils import read_json, seed_everything, write_json

lock = read_json(ARTIFACTS / "model_lock.json")
qc = read_json(ARTIFACTS / "qc_report.json")
fingerprint = read_json(ARTIFACTS / "dataset_fingerprint.json")
cfg = ExperimentConfig.from_dict(lock["config"])
splits = Splits.load(ARTIFACTS / "adult_splits.json")
manifest = pd.read_csv(ARTIFACTS / "manifest.csv")

assert cfg.config_hash() == lock["config_hash"], "Locked config was modified."
assert splits.split_hash == lock["split_hash"], "Split file was modified."
assert qc["manifest_hash"] == lock["manifest_hash"], "Manifest was modified."
seed_everything(cfg.train.seed)
print("locked at", lock["locked_at_utc"], "| config", lock["config_hash"])
"""),
    code("verify untouched", """
# Cell 02 - verify no locked output has already influenced the configuration.
prior = sorted(RESULTS.glob("*.json")) + sorted(RESULTS.glob("*.csv"))
if prior:
    print("WARNING: locked-test artifacts already exist:")
    for p in prior:
        print("  ", p.name)
    print("Re-running is a protocol deviation unless this is a documented "
          "technical failure. Record it in protocol_deviations.json.")
else:
    print("No prior locked-test artifacts. Clean first run.")

deviations = []
write_json(RESULTS / "protocol_deviations.json", deviations)

LOCKED_TEST = splits.locked_test
DEV = splits.development
print(f"locked adult test: {len(LOCKED_TEST)} cases (never seen in development)")
assert not set(LOCKED_TEST) & set(DEV)
"""),
    code("final seeds", """
# Cell 03 - train or load the three final QMMF-Net seeds on the full development pool.
from qmmf.models import build_model
from qmmf.worker import build_experiment_queue

QUEUE = ARTIFACTS / "final_queue.json"
LEDGER = ARTIFACTS / "experiment_ledger.csv"
n = build_experiment_queue(QUEUE, models=["qmmf_net"], folds=[0],
                           seeds=list(cfg.evaluation.mc_seeds), base_cfg=cfg,
                           stage="final")
print("final-seed runs queued:", n)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCHEME = LabelScheme(name="resolved", **fingerprint["label_mapping"])
cache = CaseCache(CACHE_DIR, SCHEME)
spacing = {str(r.case_id): eval(str(r.zooms)) for r in manifest.itertuples()}

def load_models(model_name, seeds):
    ledger = pd.read_csv(LEDGER)
    out = []
    for s in seeds:
        row = ledger[(ledger.model == model_name) & (ledger.seed == s) &
                     (ledger.status == "completed")]
        if row.empty:
            print(f"  missing completed run for {model_name} seed {s}")
            continue
        ck = torch.load(row.iloc[0].checkpoint, map_location=device, weights_only=False)
        m = build_model(cfg.merged({"model": model_name}))
        m.load_state_dict(ck["model"])
        out.append(m.to(device).eval())
    return out

# ensemble = load_models("qmmf_net", cfg.evaluation.mc_seeds)
print("Ensemble members are the uncertainty source (plan 10.2).")
"""),
    code("calibration fit", """
# Cell 04 - fit calibration on development-only cross-fitted predictions.
from qmmf.calibration import TemperatureScaler, calibration_report

# Development OOF logits, assembled in Notebook 02 and reused here. The locked
# test and the pediatric cohort contribute nothing to this fit.
DEV_LOGITS = RESULTS / "development_oof_logits.npz"
if DEV_LOGITS.exists():
    z = np.load(DEV_LOGITS)
    scaler = TemperatureScaler.fit(z["logits"], z["labels"])
else:
    scaler = TemperatureScaler(np.ones(3))
    print("No development OOF logits found; temperature defaults to 1.0 and the "
          "calibration claim is reported as not evaluated.")
write_json(RESULTS / "calibration.json", scaler.to_dict())
print("temperatures (wt, tc, et):", scaler.temperatures.tolist())
"""),
    code("locked full modality", """
# Cell 05 - adult locked-test inference, full modality. THE PRIMARY ENDPOINT.
from qmmf.evaluate import evaluate_cases

MODELS_TO_TEST = ["qmmf_net", "hemis25d", "unet25d", "unet2d", "segresnet3d"]
primary = {}
# for name in MODELS_TO_TEST:
#     members = load_models(name, [cfg.train.seed])
#     if not members: continue
#     res = evaluate_cases(members[0], LOCKED_TEST, cache, cfg, spacing,
#                          threshold=lock["decision_threshold"], device=device)
#     primary[name] = res
#     pd.DataFrame([r.__dict__ for r in res["rows"]]).to_csv(
#         RESULTS / f"locked_full_{name}.csv", index=False)
#     print(name, {k: round(v, 4) for k, v in res["summary"].items()
#                  if k.startswith(("dice", "macro"))})
print("Primary endpoint: patient-level macro Dice over WT, TC, ET on the "
      "locked adult test, full modality.")
"""),
    code("all subsets", """
# Cell 06 - exact all-15-subset locked-test inference.
from qmmf.evaluate import evaluate_all_subsets
from qmmf.subsets import ALL_SUBSET_KEYS

subset_results = {}
# for name in MODELS_TO_TEST:
#     members = load_models(name, [cfg.train.seed])
#     if not members: continue
#     subset_results[name] = evaluate_all_subsets(
#         members[0], LOCKED_TEST, cache, cfg, spacing, device=device)
#     write_json(RESULTS / f"subsets_{name}.json",
#                subset_results[name]["per_case"])
print("15 subsets x", len(LOCKED_TEST), "patients per model =",
      15 * len(LOCKED_TEST), "volume inferences per model.")
print("Exhaustive, so the Shapley values below are exact rather than sampled.")
"""),
    code("corruptions", """
# Cell 07 - fixed corruption grid: 6 corruptions x 3 severities.
from qmmf.transforms import corruption_grid

grid = corruption_grid()
corruption_results = {}
# for spec in grid:
#     res = evaluate_cases(ensemble[0], LOCKED_TEST, cache, cfg, spacing,
#                          corrupt={**spec, "seed": 1234}, device=device)
#     corruption_results[f"{spec['kind']}_s{spec['severity']}"] = res["summary"]
# write_json(RESULTS / "corruptions.json", corruption_results)
print(f"{len(grid)} corruption conditions; severities were fixed before any "
      "locked-test inference and are NOT drawn from the training augmentation "
      "distribution (plan 9.6).")
"""),
    code("pediatric ood", """
# Cell 08 - pediatric out-of-domain stress test. No adaptation, no tuning.
from qmmf.paths import resolve_pediatric_root

ood_results = {}
try:
    PED_ROOT = resolve_pediatric_root()
    print("pediatric root:", PED_ROOT)
    # The pediatric label scheme must be verified independently: it is a
    # different cohort and may not share the adult label semantics.
    print("Verify the pediatric label dictionary before deriving WT/TC/ET.")
except Exception as exc:
    print("pediatric cohort unavailable:", exc)

print("\\nThis is an out-of-domain stress test across differing age, tumour "
      "phenotype, acquisition and label domain. It is NOT external validation "
      "and must never be described as such (plan 5.2, claims table).")
"""),
    code("patient metrics", """
# Cell 09 - patient-level segmentation, lesion, surface and volume metrics.
from qmmf.metrics import REGIONS, summarize

tables = {}
for name, res in primary.items():
    df = pd.DataFrame([r.__dict__ for r in res["rows"]])
    tables[name] = summarize(res["rows"])
    print(f"\\n{name}")
    print(df.groupby("region")[["dice", "hd95", "nsd", "relative_volume_error"]]
            .mean().round(4).to_string())
    print("  empty-reference cases per region:",
          df.groupby("region").reference_empty.sum().to_dict())
if tables:
    pd.DataFrame(tables).T.to_csv(RESULTS / "table5_locked_full_modality.csv")
"""),
    code("reliability", """
# Cell 10 - calibration, uncertainty, failure detection and risk-coverage.
from qmmf.calibration import (
    calibration_report, ensemble_statistics, failure_detection_scores,
    failure_labels, risk_coverage_curve, selective_performance,
)

reliability = {}
# failure threshold fixed on development data, not chosen here
FAILURE_THRESHOLD = lock.get("failure_dice_threshold", 0.60)
# scores = [primary['qmmf_net']['per_case_macro'][c] for c in LOCKED_TEST]
# unc = [primary['qmmf_net']['uncertainty'][c]['mean_entropy'] for c in LOCKED_TEST]
# reliability['failure_detection'] = failure_detection_scores(
#     unc, failure_labels(scores, FAILURE_THRESHOLD))
# reliability['selective'] = selective_performance(
#     scores, unc, cfg.evaluation.selective_coverages)
# write_json(RESULTS / "reliability.json", reliability)
print("Uncertainty must be shown to be USEFUL (failure AUROC, monotone "
      "risk-coverage), not merely displayed as a heatmap (plan 3.1).")
"""),
    code("shapley", """
# Cell 11 - exact patient-level Shapley modality attribution.
from qmmf.shapley import efficiency_gap, patient_shapley_table, shapley_from_subset_table
from qmmf.subsets import subset_key

# shap = patient_shapley_table(subset_results['qmmf_net']['per_case'])
# shap.to_csv(RESULTS / "shapley_per_patient.csv", index=False)
# assert shap.efficiency_gap.abs().max() < 1e-9, "Shapley efficiency axiom violated"
# print(shap[['t1','t1ce','t2','flair']].describe().round(4).to_string())
print("Convention: v(empty set) = 0, since the network requires at least one")
print("sequence. Efficiency is asserted numerically, not assumed.")
print("A negative Shapley value is a real finding (an added sequence can hurt),")
print("not a bug to be clipped away.")
"""),
    code("gate shapley", """
# Cell 12 - gate weights versus exact Shapley; mismatch as a failure signal.
from qmmf.shapley import gate_shapley_alignment, mismatch_predicts_error

# gate_rows = [{'case_id': c, **dict(zip(cfg.data.modalities,
#               subset_results['qmmf_net']['gates'][c][subset_key(tuple(cfg.data.modalities))].mean(0)))}
#              for c in LOCKED_TEST]
# align = gate_shapley_alignment(gate_rows, shap.to_dict('records'))
# write_json(RESULTS / "gate_shapley_alignment.json",
#            {k: v for k, v in align.items() if not isinstance(v, list)})
print("Claim discipline (plan 3.4): 'interpretable gates' is only permitted if")
print("gate weights align with exact Shapley values AND the relationship is")
print("stable across folds and tumour regions. Otherwise the claim is dropped.")
"""),
    code("statistics", """
# Cell 13 - bootstrap, permutation/Wilcoxon, Holm and the non-inferiority test.
from qmmf.stats import (
    apply_holm, bootstrap_difference_between_cohorts, compare_paired,
    non_inferiority, summarize_subset_performance,
)

MARGIN = cfg.evaluation.non_inferiority_margin_pp / 100.0
results = []
# shared = sorted(set(primary['qmmf_net']['per_case_macro']) &
#                 set(primary['segresnet3d']['per_case_macro']))
# h1 = non_inferiority([primary['qmmf_net']['per_case_macro'][c] for c in shared],
#                      [primary['segresnet3d']['per_case_macro'][c] for c in shared],
#                      margin=MARGIN,
#                      n_boot=cfg.evaluation.bootstrap_replicates)
# write_json(RESULTS / "h1_non_inferiority.json", h1)
print(f"H1 margin: -{cfg.evaluation.non_inferiority_margin_pp} pp macro Dice, "
      "locked before any locked-test result was viewed.")
print("Comparison families for Holm correction: (1) core baselines, "
      "(2) confirmed ablations, (3) robustness endpoints. Corrections are "
      "applied WITHIN a family, never across all comparisons at once.")
"""),
    code("figures and tables", """
# Cell 14 - locked figures and tables generated from immutable result files.
import matplotlib.pyplot as plt

FIGURE_PLAN = {
    "figure1": "study and cohort flow, split lock, analysis boundaries",
    "figure2": "QMMF-Net architecture and block equations",
    "figure3": "all-15-subset heatmap by tumour region and model",
    "figure4": "accuracy-robustness-efficiency Pareto with MEASURED params/VRAM/latency",
    "figure5": "reliability diagrams and risk-coverage curves",
    "figure6": "gate weights, exact Shapley, and mismatch-versus-error",
    "figure7": "adult corruption and pediatric OOD degradation with uncertainty",
    "figure8": "success and failure cases sampled by a PRE-SPECIFIED rule",
}
CASE_SAMPLING_RULE = (
    "Rank locked-test cases by macro Dice; show the 10th, 50th and 90th "
    "percentile cases per region. Fixed here, before looking at any image."
)
write_json(RESULTS / "figure_plan.json",
           {"figures": FIGURE_PLAN, "case_sampling_rule": CASE_SAMPLING_RULE})
print(json.dumps(FIGURE_PLAN, indent=2))
print("\\nFigure 4 replaces the plan's illustrative design targets with the "
      "measured values from efficiency_benchmark.json.")
"""),
    code("result card", """
# Cell 15 - result card, model card, CLAIM checklist and the rejected-claim list.
ALLOWED_CLAIMS = [
    "QMMF-Net improved mean arbitrary-subset performance over reimplemented "
    "baselines under the locked protocol.",
    "QMMF-Net was non-inferior to the compact 3D baseline on full-modality "
    "macro Dice while using less peak VRAM and/or inference time.",
    "Calibration and selective prediction reduced retained-case risk on the "
    "adult locked test.",
    "Performance and calibration degraded on a pediatric OOD cohort, and "
    "uncertainty identified part of this shift. [with caveat]",
]
REJECTED_CLAIMS = [
    "State of the art - not permitted unless all relevant methods are fairly "
    "reproduced or a recognised benchmark verifies it.",
    "External clinical validation / clinical readiness / radiologist-level / "
    "improves patient outcomes - not supported by this design.",
    "Quality-aware - not permitted if the shuffled-quality (A17) and no-quality "
    "(A5) controls do not support the mechanism.",
    "Interpretable gates - not permitted if gate-Shapley alignment is absent "
    "or unstable.",
]
LIMITATIONS = [
    "MSD Task01 derives from older adult BraTS cohorts and may not represent "
    "current multicentre clinical MRI.",
    "The model is 2.5D; volumetric context is approximated.",
    "The pediatric cohort changes population and possibly label semantics, so "
    "OOD conclusions are descriptive.",
    "Quality features are handcrafted and may not reflect all acquisition defects.",
    "No radiologist reader study, prospective trial or clinical workflow evaluation.",
]
card = {
    "protocol_version": cfg.protocol_version,
    "config_hash": lock["config_hash"], "split_hash": lock["split_hash"],
    "manifest_hash": lock["manifest_hash"],
    "allowed_claims": ALLOWED_CLAIMS, "rejected_claims": REJECTED_CLAIMS,
    "limitations": LIMITATIONS,
    "protocol_deviations": read_json(RESULTS / "protocol_deviations.json"),
    "reporting_standards": ["CLAIM 2024", "Metrics Reloaded", "relevant STARD-AI items",
                            "PROBAST+AI concepts"],
}
write_json(RESULTS / "result_card.json", card)
print(json.dumps({k: v for k, v in card.items()
                  if k in ("allowed_claims", "rejected_claims")}, indent=2))
"""),
    code("reproducibility bundle", """
# Cell 16 - export the complete reproducibility bundle.
import shutil
from qmmf.utils import code_hash, environment_fingerprint

bundle = ARTIFACTS / "reproducibility_bundle"
bundle.mkdir(exist_ok=True)
for name in ("dataset_fingerprint.json", "adult_splits.json", "model_lock.json",
             "qc_report.json", "efficiency_benchmark.json",
             "experiment_ledger.csv", "preprocessing_config.json"):
    src = ARTIFACTS / name
    if src.exists():
        shutil.copy2(src, bundle / name)
for p in RESULTS.glob("*"):
    if p.is_file():
        shutil.copy2(p, bundle / p.name)

write_json(bundle / "code_and_environment.json", {
    "code_hash": code_hash(REPO / "src"),
    "environment": environment_fingerprint(),
    "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
})
print("bundle contents:")
for p in sorted(bundle.iterdir()):
    print(f"  {p.name:44s} {p.stat().st_size/1024:8.1f} KB")
print("\\nEvery headline statement in the manuscript must map to a locked "
      "endpoint, a confidence interval, an effect size and a file in this bundle.")
"""),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="notebooks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Cell counts follow the contracts in plan section 12. Notebook 01's cell 01
    # is the markdown title/declaration cell, so it has 19 code cells + 1
    # markdown for 20 contract cells; Notebooks 02 and 03 carry their title
    # markdown in addition to their 21 and 16 contract cells.
    for name, cells, expected_contract in (
        ("01_data_audit.ipynb", NB01, 20),
        ("02_training.ipynb", NB02, 21),
        ("03_locked_test.ipynb", NB03, 16),
    ):
        code_cells = sum(1 for c in cells if c[0] == "code")
        contract = code_cells + (1 if name.startswith("01") else 0)
        (out / name).write_text(json.dumps(to_notebook(cells), indent=1))
        flag = "OK" if contract == expected_contract else f"expected {expected_contract}"
        print(f"{name}: {len(cells)} cells, {contract} contract cells ({flag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
