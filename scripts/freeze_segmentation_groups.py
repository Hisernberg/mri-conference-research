#!/usr/bin/env python3
"""Freeze repaired splits from completed audits, without reading model outcomes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/src"))
from qmmf.manifest import read_manifest
from qmmf.splits import Splits, build_grouped_splits, validate_splits


def read(path):
    return Path(path).read_text()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def freeze():
    prepared = ROOT / "kaggle_outputs/preparation/prepared"
    similarity = ROOT / "kaggle_outputs/similarity_audit_v2/segmentation_similarity"
    dest = ROOT / "segmentation/configs/grouped_v2"
    dest.mkdir(parents=True, exist_ok=True)
    audit = json.loads(read(similarity / "completion.json"))
    report = json.loads(read(prepared / "dataset_fingerprint.json"))
    assert audit["completed"] and audit["correlation_dtype"] == "float64"
    assert audit["manifest_hash"] == report["manifest_hash"]
    pairs = pd.read_csv(similarity / "similarity_pairs.csv")
    values = pairs[[c for c in pairs if c.startswith("correlation_")]].values
    assert np.isfinite(values).all() and (np.abs(values) <= 1).all()
    grouping = json.loads(read(similarity / "case_groups.json"))
    mapping = grouping["case_to_group"]
    manifest = read_manifest(prepared / "manifest.csv")
    assert len(manifest) == audit["n_cases"] == len(mapping) == 484
    assert len(set(mapping.values())) == audit["n_groups"]
    for row in pairs.itertuples():
        assert mapping[row.case_a] == mapping[row.case_b]
    old = Splits.load(prepared / "splits.json")
    exposed_cases = old.folds["0"]["train"]
    exposed_groups = sorted({mapping[c] for c in exposed_cases})
    splits = build_grouped_splits(manifest, mapping, locked_ineligible_groups=exposed_groups)
    validate_splits(splits)
    # Refuse to silently replace an already frozen assignment.
    split_path = dest / "splits.json"
    if split_path.exists():
        assert Splits.load(split_path).split_hash == splits.split_hash
    splits.save(split_path)
    cohorts = {}
    counts = []
    for fold, part in splits.folds.items():
        def representatives(role):
            groups = {}
            for cid in part[role]:
                groups.setdefault(mapping[cid], []).append(cid)
            # Image-only group membership, then a fixed label-blind hash rule.
            chosen = [min(members, key=lambda c: hashlib.sha256(
                f"20260909|{fold}|{role}|{c}".encode()).hexdigest())
                for group, members in sorted(groups.items())]
            np.random.default_rng(20260909 + int(fold)).shuffle(chosen)
            return chosen
        inner, outer = representatives("inner_val"), representatives("outer_val")
        cohorts[fold] = {"validation_pilot": inner[:4], "validation_study": inner[:12],
                        "validation_all_representatives": inner,
                        "outer_evaluation": part["outer_val"],
                        "robustness_pilot": outer[:8], "robustness_study": outer[:16],
                        "outer_representatives": outer}
        for role, cases in part.items():
            counts.append({"fold": fold, "role": role, "cases": len(cases),
                           "groups": len({mapping[c] for c in cases})})
    write(dest / "development_cohorts.json", {"folds": cohorts, "locked_test_opened": False,
        "representative_rule": "minimum SHA256 of fixed salt/fold/role/case ID within each group; deterministic group shuffle",
        "cohort_seed": 20260909})
    provenance = {"manifest_hash": report["manifest_hash"], "split_hash": splits.split_hash,
        "similarity_rule": grouping["rule"], "n_cases": len(mapping), "n_groups": audit["n_groups"],
        "similarity_audit_kernel": "dasshovon/mri-segmentation-similarity-audit",
        "similarity_audit_version": 2, "similarity_source_sha256": hashlib.sha256(
            (ROOT / "scripts/audit_segmentation_similarity.py").read_bytes()).hexdigest(),
        "similarity_csv_sha256": hashlib.sha256((similarity / "similarity_pairs.csv").read_bytes()).hexdigest(),
        "group_map_sha256": hashlib.sha256((similarity / "case_groups.json").read_bytes()).hexdigest(),
        "cancelled_pilot_split_hash": old.split_hash, "pilot_exposed_cases": exposed_cases,
        "pilot_exposed_groups": exposed_groups, "unexposed_eligible_groups": len(set(mapping.values())) - len(exposed_groups),
        "locked_test_cases": len(splits.locked_test),
        "locked_test_groups": len({mapping[c] for c in splits.locked_test}),
        "test_outcomes_used": False, "fresh_weights_required": True,
        "primary_metric": "case macro Dice, averaged within image group, then equally across groups",
        "training_sampling": "uniform image group, then uniform case within group",
        "normalizer_fit": "median descriptor matrix within each training group, then robust scaler across training groups",
        "quality_shuffle": "fixed training-group derangement; donor always from a different training group"}
    write(dest / "provenance.json", provenance)
    pd.DataFrame(counts).to_csv(dest / "split_counts.csv", index=False)
    curated = ROOT / "results/segmentation_audit/grouped_v2"
    curated.mkdir(parents=True, exist_ok=True)
    for path in [*dest.glob("*"), *similarity.glob("*")]:
        if path.is_file():
            shutil.copy2(path, curated / path.name)
    print(json.dumps({k: provenance[k] for k in ["split_hash", "n_cases", "n_groups",
        "unexposed_eligible_groups", "locked_test_cases", "locked_test_groups"]}, indent=2))
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    freeze()
