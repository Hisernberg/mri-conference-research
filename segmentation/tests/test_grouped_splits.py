"""Prevent the real rescaled-copy split failure, including pilot exposure."""
import copy
import pandas as pd
import pytest
from qmmf.splits import Splits, build_grouped_splits, build_splits, validate_splits


def sample():
    rows, mapping = [], {}
    for i in range(64):
        for j in range(2):
            cid = f"case_{i:03}_{j}"
            mapping[cid] = f"group_{i:03}"
            rows.append({"case_id": cid, "wt_volume_ml": 10 + i + j / 10,
                         "et_present": int(i % 3 != 0)})
    return pd.DataFrame(rows), mapping


def test_grouped_partitions_exclude_all_related_and_pilot_exposed_cases(tmp_path):
    manifest, mapping = sample()
    exposed = sorted(set(mapping.values()))[:16]
    splits = build_grouped_splits(manifest, mapping, locked_ineligible_groups=exposed)
    validate_splits(splits)
    assert not {mapping[c] for c in splits.locked_test} & set(exposed)
    path = tmp_path / "splits.json"; splits.save(path)
    loaded = Splits.load(path)
    assert loaded.case_to_group == mapping and loaded.split_hash == splits.split_hash
    # Moving only one copy preserves case coverage, but must fail group isolation.
    leaked = copy.deepcopy(loaded)
    moved = leaked.folds["0"]["train"].pop()
    leaked.folds["0"]["inner_val"].append(moved)
    with pytest.raises(AssertionError, match="Similarity group crosses"):
        validate_splits(leaked)


def test_insufficient_unexposed_groups_fail_without_relaxing_holdout():
    manifest, mapping = sample()
    with pytest.raises(ValueError, match="Insufficient unexposed"):
        build_grouped_splits(manifest, mapping, locked_ineligible_groups=sorted(set(mapping.values()))[:-5])


def test_legacy_split_hash_remains_readable_for_provenance(tmp_path):
    manifest, _ = sample()
    original = build_splits(manifest)
    assert "case_to_group" not in original.to_dict()
    path = tmp_path / "legacy.json"; original.save(path)
    assert Splits.load(path).split_hash == original.split_hash
