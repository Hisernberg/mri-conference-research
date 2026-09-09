"""Hashed case and conservative-group splits, with legacy provenance support.

`build_splits` preserves the original case-ID protocol for auditing. Real MRI
experiments use `build_grouped_splits` to keep related volumes in the same role.
Similarity groups do not establish verified patient identities.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import sha256_obj, write_json


LOCKED_TEST_FRACTION = 0.20
N_FOLDS = 4
INNER_VAL_FRACTION = 0.15   # of each training fold, for early stop / thresholds
LABEL_FRACTIONS = (0.25, 0.50, 1.00)


@dataclass
class Splits:
    locked_test: List[str]
    development: List[str]
    folds: Dict[str, Dict[str, List[str]]]      # "0" -> {train, inner_val, outer_val}
    label_subsets: Dict[str, List[str]]         # "0.25" -> case ids
    strata: Dict[str, str]
    seed: int
    split_hash: str = ""
    case_to_group: Dict[str, str] = field(default_factory=dict)
    partition_policy: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("split_hash", None)
        # Keep old hashes readable as provenance, while hashing every new group.
        if not d["case_to_group"]:
            d.pop("case_to_group")
        if not d["partition_policy"]:
            d.pop("partition_policy")
        return d

    def compute_hash(self) -> str:
        return sha256_obj(self.to_dict())[:16]

    def save(self, path: str | Path) -> str:
        self.split_hash = self.compute_hash()
        payload = self.to_dict()
        payload["split_hash"] = self.split_hash
        write_json(path, payload)
        return self.split_hash

    @classmethod
    def load(cls, path: str | Path) -> "Splits":
        import json
        payload = json.loads(Path(path).read_text())
        stored = payload.pop("split_hash", "")
        obj = cls(**payload)
        recomputed = obj.compute_hash()
        if stored and stored != recomputed:
            raise ValueError(
                f"Split file hash mismatch: stored {stored}, recomputed {recomputed}. "
                "The partition has been modified; refuse to train or test."
            )
        obj.split_hash = recomputed
        return obj


# --------------------------------------------------------------------------- #
def make_strata(
    manifest: pd.DataFrame,
    volume_col: str = "wt_volume_ml",
    et_col: str = "et_present",
    n_quantiles: int = 4,
    min_per_stratum: int = 8,
) -> pd.Series:
    """Stratification code = ET presence x tumour-volume quartile (plan 6.1).

    Sparse strata are merged deterministically so that every stratum can supply
    at least one case to the locked test and to each of the four folds.
    """
    if volume_col not in manifest or et_col not in manifest:
        raise KeyError(f"Manifest needs '{volume_col}' and '{et_col}' columns.")
    vol = manifest[volume_col].astype(float)
    try:
        quart = pd.qcut(vol, q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        quart = pd.Series(np.zeros(len(vol), dtype=int), index=manifest.index)
    quart = quart.fillna(0).astype(int)
    et = manifest[et_col].astype(int)
    code = et.astype(str) + "_q" + quart.astype(str)

    counts = code.value_counts()
    # Merge any stratum that is too small into the nearest quartile with the
    # same ET status; fall back to the ET status alone.
    remap: Dict[str, str] = {}
    for key, count in counts.items():
        if count >= min_per_stratum:
            continue
        et_val = key.split("_")[0]
        siblings = [k for k, c in counts.items()
                    if k != key and k.startswith(et_val + "_") and c >= min_per_stratum]
        remap[key] = siblings[0] if siblings else f"{et_val}_merged"
    return code.map(lambda k: remap.get(k, k))


def _stratified_partition(
    ids: Sequence[str], strata: Sequence[str], fractions: Sequence[float],
    rng: np.random.Generator,
) -> List[List[str]]:
    """Split ids into len(fractions) groups, stratified and deterministic."""
    ids = list(ids)
    strata = list(strata)
    groups: List[List[str]] = [[] for _ in fractions]
    by_stratum: Dict[str, List[str]] = {}
    for cid, s in zip(ids, strata):
        by_stratum.setdefault(s, []).append(cid)

    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        perm = rng.permutation(len(members))
        members = [members[i] for i in perm]
        n = len(members)
        # Largest-remainder allocation keeps the totals exact.
        raw = [f * n for f in fractions]
        counts = [int(np.floor(r)) for r in raw]
        remainder = n - sum(counts)
        order = np.argsort([-(r - np.floor(r)) for r in raw], kind="stable")
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1
        pos = 0
        for gi, c in enumerate(counts):
            groups[gi].extend(members[pos:pos + c])
            pos += c
    return [sorted(g) for g in groups]


def build_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    case_col: str = "case_id",
    locked_fraction: float = LOCKED_TEST_FRACTION,
    n_folds: int = N_FOLDS,
    inner_val_fraction: float = INNER_VAL_FRACTION,
) -> Splits:
    """Create the locked test, the four development folds and nested label subsets.

    Returns a Splits object; call .save() to freeze it with a hash.
    """
    if manifest[case_col].duplicated().any():
        raise ValueError("Duplicate case ids in manifest; resolve before splitting.")
    rng = np.random.default_rng(seed)
    strata = make_strata(manifest)
    ids = manifest[case_col].astype(str).tolist()
    strata_map = {cid: str(s) for cid, s in zip(ids, strata)}

    locked, development = _stratified_partition(
        ids, [strata_map[c] for c in ids], [locked_fraction, 1 - locked_fraction], rng
    )

    # Four outer-validation partitions over the development pool.
    fold_parts = _stratified_partition(
        development, [strata_map[c] for c in development],
        [1.0 / n_folds] * n_folds, rng,
    )

    folds: Dict[str, Dict[str, List[str]]] = {}
    for k in range(n_folds):
        outer_val = fold_parts[k]
        train_pool = sorted(set(development) - set(outer_val))
        inner_val, train = _stratified_partition(
            train_pool, [strata_map[c] for c in train_pool],
            [inner_val_fraction, 1 - inner_val_fraction],
            np.random.default_rng(seed + 1000 + k),
        )
        folds[str(k)] = {
            "train": sorted(train),
            "inner_val": sorted(inner_val),
            "outer_val": sorted(outer_val),
        }

    label_subsets = build_nested_label_subsets(
        development, strata_map, seed=seed, fractions=LABEL_FRACTIONS
    )

    splits = Splits(
        locked_test=sorted(locked),
        development=sorted(development),
        folds=folds,
        label_subsets=label_subsets,
        strata=strata_map,
        seed=seed,
    )
    validate_splits(splits)
    splits.split_hash = splits.compute_hash()
    return splits


def build_grouped_splits(
    manifest: pd.DataFrame,
    case_to_group: Dict[str, str],
    seed: int = 42,
    locked_ineligible_groups: Sequence[str] = (),
    locked_fraction: float = LOCKED_TEST_FRACTION,
    n_folds: int = N_FOLDS,
    inner_val_fraction: float = INNER_VAL_FRACTION,
) -> Splits:
    """Partition complete similarity groups; never split related case IDs.

    A fresh protected test can exclude groups exposed during an aborted pilot.
    Stratification uses median group tumour volume and any enhancing target.
    """
    frame = manifest.copy()
    ids = frame.case_id.astype(str).tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(case_to_group):
        raise ValueError("Grouping must cover the unique manifest case IDs exactly")
    frame["group_id"] = frame.case_id.map(case_to_group)
    frame["wt_volume_ml"] = frame.wt_volume_ml.astype(float)
    frame["et_present"] = frame.et_present.astype(int)
    grouped = frame.groupby("group_id", sort=True).agg(
        wt_volume_ml=("wt_volume_ml", "median"), et_present=("et_present", "max"))
    grouped = grouped.reset_index().rename(columns={"group_id": "case_id"})
    group_ids = grouped.case_id.tolist()
    strata = make_strata(grouped)
    strata_map = dict(zip(group_ids, map(str, strata)))
    excluded = set(locked_ineligible_groups)
    if not excluded <= set(group_ids):
        raise ValueError("Unknown group in protected-test exclusion list")
    eligible = sorted(set(group_ids) - excluded)
    target = max(1, int(round(locked_fraction * len(group_ids))))
    if len(eligible) <= target:
        raise ValueError("Insufficient unexposed groups for a fresh protected test")
    fraction = target / len(eligible)
    locked, _ = _stratified_partition(
        eligible, [strata_map[g] for g in eligible], [fraction, 1 - fraction],
        np.random.default_rng(seed))
    development = sorted(set(group_ids) - set(locked))
    fold_parts = _stratified_partition(
        development, [strata_map[g] for g in development], [1 / n_folds] * n_folds,
        np.random.default_rng(seed + 1))
    by_group = frame.groupby("group_id").case_id.agg(list).to_dict()
    def expand(groups):
        return sorted(cid for group in groups for cid in by_group[group])
    folds = {}
    for k, outer in enumerate(fold_parts):
        pool = sorted(set(development) - set(outer))
        inner, train = _stratified_partition(
            pool, [strata_map[g] for g in pool], [inner_val_fraction, 1 - inner_val_fraction],
            np.random.default_rng(seed + 1000 + k))
        folds[str(k)] = {"train": expand(train), "inner_val": expand(inner), "outer_val": expand(outer)}
    label_groups = build_nested_label_subsets(development, strata_map, seed=seed)
    result = Splits(locked_test=expand(locked), development=expand(development), folds=folds,
        label_subsets={key: expand(value) for key, value in label_groups.items()},
        strata={cid: strata_map[case_to_group[cid]] for cid in sorted(ids)}, seed=seed,
        case_to_group=dict(sorted(case_to_group.items())),
        partition_policy={"unit": "conservative_image_similarity_group",
            "locked_ineligible_groups": sorted(excluded), "eligible_test_groups": len(eligible),
            "target_test_groups": target, "actual_test_groups": len(locked),
            "group_stratification": "median WT volume and any ET presence",
            "pilot_exposure_excluded": bool(excluded)})
    validate_splits(result)
    result.split_hash = result.compute_hash()
    return result


def build_nested_label_subsets(
    development: Sequence[str],
    strata_map: Dict[str, str],
    seed: int = 42,
    fractions: Sequence[float] = LABEL_FRACTIONS,
) -> Dict[str, List[str]]:
    """Strictly nested 25% subset of 50% subset of 100% (plan 6.5)."""
    fractions = sorted(fractions)
    rng = np.random.default_rng(seed + 77)
    current = sorted(development)
    subsets: Dict[str, List[str]] = {f"{max(fractions):.2f}": sorted(development)}
    # Build from the largest downwards so each is a strict subset of the previous.
    for frac in sorted(fractions, reverse=True)[1:]:
        target_n = int(round(frac * len(development)))
        keep, _ = _stratified_partition(
            current, [strata_map[c] for c in current],
            [target_n / len(current), 1 - target_n / len(current)], rng,
        )
        subsets[f"{frac:.2f}"] = sorted(keep)
        current = sorted(keep)
    return dict(sorted(subsets.items()))


# --------------------------------------------------------------------------- #
def validate_splits(splits: Splits) -> None:
    """Assertions the plan requires before training is permitted (gate G0)."""
    locked = set(splits.locked_test)
    dev = set(splits.development)
    if locked & dev:
        raise AssertionError(f"Locked test leaks into development: {sorted(locked & dev)}")
    if not locked or not dev:
        raise AssertionError("Empty locked test or development pool.")

    seen_outer: List[str] = []
    for k, part in splits.folds.items():
        tr, iv, ov = set(part["train"]), set(part["inner_val"]), set(part["outer_val"])
        if tr & iv or tr & ov or iv & ov:
            raise AssertionError(f"Fold {k} partitions overlap.")
        if (tr | iv | ov) != dev:
            missing = dev - (tr | iv | ov)
            extra = (tr | iv | ov) - dev
            raise AssertionError(
                f"Fold {k} does not cover the development pool exactly "
                f"(missing {len(missing)}, extra {len(extra)})."
            )
        if (tr | iv | ov) & locked:
            raise AssertionError(f"Fold {k} touches the locked test.")
        seen_outer.extend(part["outer_val"])

    # Each development case appears once in the outer partitions. Patient
    # independence additionally needs the group/identity checks below.
    counts = pd.Series(seen_outer).value_counts()
    if len(seen_outer) != len(dev) or (counts != 1).any():
        raise AssertionError("Outer-validation partitions do not cover each case once.")

    keys = sorted(splits.label_subsets, key=float)
    for small, large in zip(keys, keys[1:]):
        if not set(splits.label_subsets[small]) <= set(splits.label_subsets[large]):
            raise AssertionError(
                f"Label subset {small} is not a strict subset of {large}."
            )
    if not set(splits.label_subsets[keys[-1]]) <= dev:
        raise AssertionError("Label subsets contain non-development cases.")

    if splits.case_to_group:
        mapping = splits.case_to_group
        if set(mapping) != locked | dev or any(not isinstance(g, str) or not g for g in mapping.values()):
            raise AssertionError("Group map must cover every case exactly")
        def groups(cases):
            return {mapping[c] for c in cases}
        locked_groups = groups(locked)
        if locked_groups & groups(dev):
            raise AssertionError("Similarity group crosses locked test and development")
        forbidden = set(splits.partition_policy.get("locked_ineligible_groups", []))
        if locked_groups & forbidden:
            raise AssertionError("Protected test includes a group exposed during the aborted pilot")
        for k, part in splits.folds.items():
            roles = {name: groups(cases) for name, cases in part.items()}
            names = list(roles)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if roles[a] & roles[b]:
                        raise AssertionError(f"Similarity group crosses {a}/{b} in fold {k}")
        for key, cases in splits.label_subsets.items():
            selected = set(cases)
            for cid in dev - selected:
                if mapping[cid] in groups(selected):
                    raise AssertionError(f"Label subset {key} splits a similarity group")


def training_cases(
    splits: Splits, fold: int, label_fraction: float = 1.0
) -> List[str]:
    """Training ids for a fold, intersected with the nested label subset."""
    part = splits.folds[str(fold)]
    allowed = set(splits.label_subsets[f"{label_fraction:.2f}"])
    return sorted(set(part["train"]) & allowed)
