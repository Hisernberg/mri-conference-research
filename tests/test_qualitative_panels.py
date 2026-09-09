"""Synthetic-only tests for figure geometry and quantitative provenance."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
import qualitative_panels as panels
import analyze_qualitative as review
from qmmf.splits import Splits


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2))


def test_axial_ties_empty_fallback_and_ras_orientation():
    reference = np.zeros((3, 2, 3, 5), dtype=np.uint8)
    reference[0, 0, 0, 1:3] = 1
    assert panels.axial_index(reference) == 1
    assert panels.axial_index(np.zeros_like(reference)) == 2
    masks = np.zeros((3, 2, 3), dtype=bool)
    masks[0, 0, 0] = True          # left/posterior -> bottom left
    masks[:, 1, 2] = True          # right/anterior -> top right
    policy = panels.read_policy()
    rgb = panels.overlay_rgb(np.zeros((2, 3)), np.ones((2, 3)), masks, policy)
    assert rgb.shape == (3, 2, 3)
    assert rgb[-1, 0].argmax() == 1
    assert rgb[0, -1].argmax() == 0
    assert review.palette_region_counts(rgb) == {"wt": 2, "tc": 1, "et": 1}
    # A false-positive region outside brain support must stay visible.
    rgb = panels.overlay_rgb(np.zeros((2, 3)), np.zeros((2, 3)), masks, policy)
    assert review.palette_region_counts(rgb)["wt"] == 2


def test_rejects_nonfinite_or_non_nested_panels():
    masks = np.zeros((3, 2, 3), dtype=bool)
    policy = panels.read_policy()
    with pytest.raises(ValueError, match="finite-value"):
        panels.overlay_rgb(np.full((2, 3), np.nan), np.ones((2, 3)), masks, policy)
    masks[2, 0, 0] = True
    with pytest.raises(ValueError, match="nesting"):
        panels.overlay_rgb(np.zeros((2, 3)), np.ones((2, 3)), masks, policy)


@pytest.fixture
def example_source(tmp_path):
    policy = panels.read_policy()
    quant = json.loads((ROOT / "segmentation/configs/grouped_v2/locked_evaluation_protocol.json").read_text())
    splits = Splits.load(ROOT / "segmentation/configs/grouped_v2/splits.json")
    source = tmp_path / "source"
    write(source / "qualitative_protocol.json", policy)
    gate = {"qualitative_protocol": policy, "qualitative_protocol_sha256": review.json_sha(policy),
            "frozen_protocol_sha256": policy["quantitative_protocol_sha256"]}
    image = np.linspace(-2, 2, 7*9*5).reshape(7, 9, 5).astype(np.float16)
    brain = np.ones_like(image, dtype=bool); brain[0] = False
    reference = np.zeros((3, 7, 9, 5), dtype=np.uint8)
    reference[0, 2:5, 3:7, 1:3] = 1
    reference[1, 3:5, 4:6, 1:3] = 1
    reference[2, 4:5, 5:6, 1] = 1
    indexed = {}
    for index, name in enumerate(policy["models"]):
        run = source / "runs" / f"{name}_s42"
        prediction = np.roll(reference, index % 2, axis=1).astype(bool)
        prediction[0, 0, 8, 1] = True
        checkpoint = hashlib.sha256(name.encode()).hexdigest()
        entries, rows = [], []
        for cid in policy["case_ids"]:
            record = panels.save_case_tiles(cid, image, brain, reference, prediction,
                                           run / "qualitative", policy)
            dice = {region: float(2*np.logical_and(reference[k], prediction[k]).sum() /
                                  (reference[k].sum()+prediction[k].sum()))
                    for k, region in enumerate(("wt", "tc", "et"))}
            record.update(group_id=splits.case_to_group[cid], whole_volume_region_dice=dice,
                          whole_volume_macro_dice=float(np.mean(list(dice.values()))))
            entries.append(record)
            rows += [{"case_id": cid, "region": region, "dice": value, "reference_empty": False}
                     for region, value in dice.items()]
        write(run / "qualitative_manifest.json", {"model": name, "seed": 42,
            "checkpoint_sha256": checkpoint, "qualitative_protocol_sha256": review.json_sha(policy),
            "case_ids": policy["case_ids"], "panels": entries, "training_performed": False})
        indexed[(42, name)] = {"done": {"checkpoint_sha256": checkpoint, "qualitative_cases": 3},
                               "case_metrics": pd.DataFrame(rows)}
    return source, tmp_path / "review", quant, splits, gate, indexed


def test_complete_prespecified_examples_render_with_attribution(example_source):
    result = review.verify_and_plot(*example_source, synthetic=True)
    assert result["verified"] and result["prediction_panels"] == 12
    assert result["synthetic_fixture"]
    assert (example_source[1] / "prespecified_examples.png").stat().st_size > 1000
    assert (example_source[1] / "prespecified_examples.pdf").stat().st_size > 1000
    notice = (example_source[1] / "ATTRIBUTION.md").read_text()
    assert "CC BY-SA 4.0" in notice and "SYNTHETIC TEST FIXTURE" in notice


@pytest.mark.parametrize("damage", ["slice", "score", "checkpoint", "input", "png", "missing_model", "seed", "case", "region"])
def test_rejects_changed_qualitative_evidence(example_source, damage):
    source = example_source[0]
    path = source / "runs/hemis_s42/qualitative_manifest.json"
    data = json.loads(path.read_text()); panel = data["panels"][0]
    if damage == "slice":
        panel["axial_index"] += 1
    elif damage == "score":
        panel["whole_volume_macro_dice"] += .01
    elif damage == "checkpoint":
        data["checkpoint_sha256"] = "b"*64
    elif damage == "input":
        panel["image_slice_sha256"] = "b"*64
    elif damage == "seed":
        data["seed"] = 43
    elif damage == "case":
        panel["case_id"] = data["panels"][1]["case_id"]
    elif damage == "region":
        del panel["whole_volume_region_dice"]["et"]
    elif damage == "png":
        png = path.parent / panel["prediction_png"]
        with Image.open(png) as im:
            pixels = np.array(im)
        pixels[0, 0] = [255, 0, 0]
        Image.fromarray(pixels).save(png)
    if damage == "missing_model":
        path.unlink()
    else:
        write(path, data)
    with pytest.raises(ValueError):
        review.verify_and_plot(*example_source)
