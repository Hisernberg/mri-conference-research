"""Reject mismatched or incomplete cache attachments before expensive GPU fitting."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("conference_runner",
    Path(__file__).resolve().parents[1] / "segmentation/scripts/conference_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_training_cache_requires_matching_inventory_and_every_archive(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    for cid in ["a", "b"]:
        (cache / f"{cid}.npz").touch()
    inventory = tmp_path / "case_inventory.csv"
    pd.DataFrame({"case_id": ["a", "b"]}).to_csv(inventory, index=False)
    complete = {"completed": True, "voxel_equality_verified_for_all_cases": True,
        "manifest_hash": "known", "case_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest()}
    (tmp_path / "training_slabs_complete.json").write_text(json.dumps(complete))
    splits = SimpleNamespace(case_to_group={"a": "g1", "b": "g2"})
    assert runner.find_training_slabs(tmp_path, {"manifest_hash": "known"}, splits)[0] == cache
    with pytest.raises(ValueError, match="different dataset"):
        runner.find_training_slabs(tmp_path, {"manifest_hash": "wrong"}, splits)
    (cache / "b.npz").unlink()
    with pytest.raises(ValueError, match="missing"):
        runner.find_training_slabs(tmp_path, {"manifest_hash": "known"}, splits)
    inventory.write_text("case_id\na\n")
    with pytest.raises(ValueError, match="inventory hash mismatch"):
        runner.find_training_slabs(tmp_path, {"manifest_hash": "known"}, splits)
