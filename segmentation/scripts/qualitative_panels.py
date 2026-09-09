"""Prespecified, attributed segmentation figures; no fitting or case ranking."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "audit/qualitative_protocol.json"


def read_policy(quantitative_protocol=None):
    policy = json.loads(POLICY_PATH.read_text())
    if quantitative_protocol is not None:
        candidates = quantitative_protocol["robustness_case_ids"]
        chosen = sorted(candidates, key=lambda c: hashlib.sha256(
            (policy["case_selection_salt"] + c).encode()).hexdigest())[:3]
        if policy["case_ids"] != chosen or policy["split_hash"] != quantitative_protocol["split_hash"]:
            raise ValueError("Qualitative cases differ from their frozen hash selection")
    if policy["seed"] != 42 or policy["case_count"] != 3 or policy["threshold"] != .5:
        raise ValueError("Qualitative seed, cohort size or threshold changed")
    return policy


def array_sha(array):
    array = np.ascontiguousarray(array)
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def axial_index(reference):
    reference = np.asarray(reference)
    if reference.ndim != 4 or reference.shape[0] != 3:
        raise ValueError("Expected WT/TC/ET reference in original volume geometry")
    areas = reference[0].astype(bool).sum(axis=(0, 1))
    return int(np.argmax(areas)) if areas.max() > 0 else reference.shape[-1] // 2


def region_labels(regions):
    regions = np.asarray(regions, dtype=bool)
    if regions.ndim != 3 or regions.shape[0] != 3:
        raise ValueError("Expected three two-dimensional region masks")
    if np.any(regions[1] & ~regions[0]) or np.any(regions[2] & ~regions[1]):
        raise ValueError("Qualitative region masks violate WT/TC/ET nesting")
    labels = np.zeros(regions.shape[1:], dtype=np.uint8)
    for label, mask in enumerate(regions, 1):
        labels[mask] = label
    return labels


def overlay_rgb(image, brain_mask, regions, policy):
    """Return an RGB axial tile with A at top and R at viewer right."""
    image = np.asarray(image, dtype=np.float32)
    brain_mask = np.asarray(brain_mask, dtype=bool)
    labels = region_labels(regions)
    if image.shape != labels.shape or brain_mask.shape != image.shape or not np.isfinite(image).all():
        raise ValueError("Qualitative image/mask geometry or finite-value check failed")
    low, high = policy["display_window"]
    if not high > low:
        raise ValueError("Invalid fixed display window")
    gray = np.clip((image - low) / (high - low), 0, 1)
    gray[~brain_mask] = 0
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    alpha = policy["overlay_alpha"]
    for label, color in enumerate(policy["colors_rgb"], 1):
        hit = labels == label
        rgb[hit] = (1 - alpha) * rgb[hit] + alpha * np.asarray(color)
    # Source axes are X(R), Y(A). PNG rows run from top to bottom.
    return np.rint(255 * rgb.transpose(1, 0, 2)[::-1]).astype(np.uint8)


def save_case_tiles(case_id, image, brain_mask, reference, prediction, dest, policy):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("Case identifier is not a safe artifact name")
    image, brain_mask = np.asarray(image), np.asarray(brain_mask)
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.shape != (3, *image.shape) or brain_mask.shape != image.shape:
        raise ValueError("Qualitative volumes do not share original geometry")
    z = axial_index(reference)
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    record = {"case_id": case_id, "original_shape": list(image.shape), "axial_index": z,
        "whole_tumor_empty": bool(not reference[0].any()),
        "reference_axial_whole_tumor_areas": reference[0].astype(bool).sum(axis=(0, 1)).tolist(),
        "reference_slice_region_pixels": dict(zip(("wt", "tc", "et"),
            reference[..., z].astype(bool).sum(axis=(1, 2)).tolist())),
        "prediction_slice_region_pixels": dict(zip(("wt", "tc", "et"),
            prediction[..., z].astype(bool).sum(axis=(1, 2)).tolist())),
        "image_slice_sha256": array_sha(image[..., z]),
        "brain_slice_sha256": array_sha(brain_mask[..., z]),
        "reference_slice_sha256": array_sha(reference[..., z]),
        "prediction_slice_sha256": array_sha(prediction[..., z]), "orientation": policy["orientation"]}
    for role, regions in [("reference", reference), ("prediction", prediction)]:
        filename = f"{case_id}_{role}.png"
        pixels = overlay_rgb(image[..., z], brain_mask[..., z], regions[..., z], policy)
        Image.fromarray(pixels).save(dest / filename)
        record[role + "_png"] = "qualitative/" + filename
        record[role + "_png_sha256"] = hashlib.sha256((dest / filename).read_bytes()).hexdigest()
    return record


def save_model_panels(model, cache, cfg, spacing, quantitative_protocol, full_summary,
                      name, seed, checkpoint_sha256, splits, dest, device):
    """Repeat inference only for three fixed cases with the seed-42 checkpoint."""
    from qmmf.inference import predict_volume, binarize
    from qmmf.metrics import evaluate_case, macro_dice
    from qmmf.transforms import uncrop_to_original
    policy = read_policy(quantitative_protocol)
    if seed != policy["seed"]:
        return None
    dest = Path(dest)
    records = []
    for cid in policy["case_ids"]:
        out = predict_volume(model, cid, cache, cfg, device=device)
        reference = out["reference"]
        prediction = binarize(out["probs"], policy["threshold"])
        rows = evaluate_case(prediction, reference, spacing[cid], case_id=cid,
                             surface_metrics=False, lesion_metrics=False)
        score = macro_dice(rows)
        if not np.isclose(score, full_summary["per_case_macro"][cid], rtol=1e-6, atol=1e-7):
            raise ValueError("Qualitative prediction differs from quantitative full-volume inference")
        cached = cache.load(cid)
        index = list(cfg.data.modalities).index(policy["display_modality"])
        image = uncrop_to_original(cached["image"][index], cached["crop_box"], cached["original_shape"])
        brain = uncrop_to_original(cached["brain_mask"], cached["crop_box"], cached["original_shape"])
        record = save_case_tiles(cid, image, brain, reference, prediction, dest / "qualitative", policy)
        record.update({"group_id": splits.case_to_group[cid], "whole_volume_macro_dice": float(score),
                       "whole_volume_region_dice": {row.region: float(row.dice) for row in rows}})
        records.append(record)
    manifest = {"model": name, "seed": seed, "checkpoint_sha256": checkpoint_sha256,
                "qualitative_protocol_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
                "case_ids": policy["case_ids"], "panels": records, "training_performed": False}
    (dest / "qualitative_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest
