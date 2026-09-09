"""Verify fixed qualitative examples against checked quantitative artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
from qualitative_panels import POLICY_PATH


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()).hexdigest()


def palette_region_counts(rgb):
    rgb = np.asarray(rgb)
    require(rgb.ndim == 3 and rgb.shape[2] == 3, "Panel is not an RGB image")
    colored = np.ptp(rgb.astype(np.int16), axis=-1) > 1
    channel = rgb.argmax(axis=-1)
    return {"wt": int(colored.sum()), "tc": int((colored & (channel != 1)).sum()),
            "et": int((colored & (channel == 0)).sum())}


def verify_and_plot(source, dest, quantitative_protocol, splits, gate, indexed, *, synthetic=False):
    source, dest = Path(source), Path(dest)
    policy = read(source / "qualitative_protocol.json")
    require(policy == read(POLICY_PATH) == gate["qualitative_protocol"],
            "Qualitative policy differs from its frozen source or development gate")
    require(json_sha(policy) == gate["qualitative_protocol_sha256"], "Qualitative policy hash differs")
    require(policy["quantitative_protocol_sha256"] == gate["frozen_protocol_sha256"],
            "Qualitative figures refer to a different quantitative protocol")
    selected = sorted(quantitative_protocol["robustness_case_ids"], key=lambda c: hashlib.sha256(
        (policy["case_selection_salt"] + c).encode()).hexdigest())[:3]
    require(selected == policy["case_ids"] and policy["seed"] == 42,
            "Qualitative case or seed selection differs")
    require(len({splits.case_to_group[c] for c in selected}) == 3,
            "Qualitative cases are not from three distinct protected groups")
    expected = {(42, name) for name in policy["models"]}
    manifests = {}
    images, case_info, rows = {}, {}, []
    for path in (source / "runs").rglob("qualitative_manifest.json"):
        manifest = read(path); identity = (manifest["seed"], manifest["model"])
        require(identity in expected and identity not in manifests, "Unexpected or repeated qualitative checkpoint")
        require(identity in indexed, "Qualitative checkpoint lacks quantitative verification")
        checked = indexed[identity]
        require(manifest["checkpoint_sha256"] == checked["done"]["checkpoint_sha256"],
                "Qualitative checkpoint differs from the quantified checkpoint")
        require(not manifest["training_performed"] and manifest["case_ids"] == selected and
                manifest["qualitative_protocol_sha256"] == gate["qualitative_protocol_sha256"],
                "Qualitative provenance or cohort differs")
        require([panel["case_id"] for panel in manifest["panels"]] == selected,
                "Qualitative panel list is incomplete or reordered")
        require(checked["done"]["qualitative_cases"] == 3, "Qualitative completion count differs")
        frame = checked["case_metrics"]
        for panel in manifest["panels"]:
            cid = panel["case_id"]
            require(panel["group_id"] == splits.case_to_group[cid] and
                    panel["orientation"] == policy["orientation"], "Qualitative group or orientation differs")
            area = np.asarray(panel["reference_axial_whole_tumor_areas"])
            require(len(area) == panel["original_shape"][2] and (area >= 0).all(), "Invalid axial reference areas")
            z = int(area.argmax()) if area.max() else len(area) // 2
            require(panel["axial_index"] == z and bool(panel["whole_tumor_empty"]) == bool(area.max() == 0),
                    "Qualitative slice violates the fixed reference-area rule")
            require(panel["reference_slice_region_pixels"]["wt"] == int(area[z]),
                    "Reference panel area differs from the selected slice")
            case = frame[frame.case_id == cid]
            macro = float(case.loc[~case.reference_empty, "dice"].mean())
            require(np.isclose(macro, panel["whole_volume_macro_dice"], rtol=1e-6, atol=1e-7),
                    "Qualitative Dice caption differs from the quantitative case score")
            require(set(panel["whole_volume_region_dice"]) == {"wt", "tc", "et"},
                    "Qualitative regional Dice list is incomplete")
            for region, value in panel["whole_volume_region_dice"].items():
                require(np.isclose(float(case.set_index("region").loc[region, "dice"]), value,
                                   rtol=1e-6, atol=1e-7), "Qualitative regional Dice differs")
            common = {key: panel[key] for key in ("original_shape", "axial_index", "image_slice_sha256",
                "brain_slice_sha256", "reference_slice_sha256", "reference_png_sha256",
                "reference_axial_whole_tumor_areas", "reference_slice_region_pixels", "orientation")}
            require(cid not in case_info or case_info[cid] == common,
                    "Models do not share the same qualitative image, reference and slice")
            case_info[cid] = common
            for role in ("reference", "prediction"):
                require(panel[role + "_png"] == f"qualitative/{cid}_{role}.png", "Unexpected panel artifact path")
                png = path.parent / panel[role + "_png"]
                require(png.is_file() and hashlib.sha256(png.read_bytes()).hexdigest() == panel[role + "_png_sha256"],
                        "Qualitative PNG is missing or its checksum differs")
                with Image.open(png) as im:
                    require(im.mode == "RGB" and list(im.size) == panel["original_shape"][:2],
                            "Qualitative PNG dimensions or mode differ")
                    pixels = np.array(im)
                require(palette_region_counts(pixels) == panel[role + "_slice_region_pixels"],
                        "Rendered panel region areas differ from its mask record")
                images[(cid, "reference" if role == "reference" else manifest["model"])] = pixels
            rows.append({"case_id": cid, "model": manifest["model"], "seed": 42,
                         "axial_index": z, "whole_volume_macro_dice": macro})
        manifests[identity] = manifest
    require(set(manifests) == expected, "Qualitative model matrix is incomplete")
    dest.mkdir(parents=True, exist_ok=True)
    render(dest, policy, images, case_info, rows, synthetic=synthetic)
    result = {"verified": True, "cases": selected, "groups": 3, "seed": 42,
        "models": policy["models"], "prediction_panels": 12, "synthetic_fixture": synthetic,
        "qualitative_protocol_sha256": gate["qualitative_protocol_sha256"],
        "interpretation": "Three prespecified examples; whole-cohort quantitative results determine population conclusions.",
        "checks": ["frozen case/seed rule", "checkpoint identity", "reference-area slice rule", "common input/reference",
                   "whole-volume Dice captions", "PNG geometry and checksums", "rendered region areas"]}
    (dest / "verification.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    (dest / "panel_metrics.json").write_text(json.dumps(rows, sort_keys=True, indent=2))
    return result


def render(dest, policy, images, case_info, rows, *, synthetic=False):
    titles = {"qmmf": "QMMF", "hemis": "HeMIS-style", "no_quality": "No quality", "unet25d": "U-Net 2.5D"}
    columns = ["reference"] + policy["models"]
    scores = {(r["case_id"], r["model"]): r["whole_volume_macro_dice"] for r in rows}
    fig, axes = plt.subplots(3, 5, figsize=(15, 10.8))
    for i, cid in enumerate(policy["case_ids"]):
        for j, name in enumerate(columns):
            ax = axes[i, j]; ax.imshow(images[(cid, name)], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title("Reference" if name == "reference" else titles[name], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cid}\naxial z={case_info[cid]['axial_index']}", fontsize=10)
            if name != "reference":
                ax.set_xlabel(f"Volume macro Dice {scores[(cid, name)]:.3f}", fontsize=9)
    title = ("SYNTHETIC TEST FIXTURE | artificial arrays and scores" if synthetic else
             "Prespecified reserved examples | full modalities | training seed 42")
    fig.suptitle(title, fontsize=13)
    fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(policy["colors_rgb"], policy["regions"])],
               loc="lower center", bbox_to_anchor=(.5, .045), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .019, "Hash-selected cases; maximum-reference-WT-area axial slices. R at viewer right; A at top.",
             ha="center", fontsize=8)
    footer = ("Synthetic arrays only. Dataset identifiers exercise cohort checks; these are not MRI experiment results." if synthetic else
              "Data: MSD/BraTS contributors, CC BY-SA 4.0. Normalized FLAIR with derived reference/prediction overlays.")
    fig.text(.5, .002, footer,
             ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .07, 1, .97), h_pad=2.8, w_pad=1.0)
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"prespecified_examples.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    attribution = (
        "# Attribution for derived qualitative figures\n\n"
        "Data: Medical Segmentation Decathlon Task01_BrainTumour, a BraTS-derived cohort, supplied by the MSD/BraTS contributors. "
        f"[Dataset mirror]({policy['dataset_url']}); [Antonelli et al., The Medical Segmentation Decathlon, 2022]({policy['scientific_source']}).\n\n"
        f"The source authors identify the dataset license as [CC BY-SA 4.0]({policy['license_url']}). These derived panels retain that license. "
        "This notice applies to the data-derived figures; it does not assign a license to the research code.\n\n"
        "Changes: canonical MRI preprocessing, normalized FLAIR display with a fixed [-3, 3] window, brain-support background masking, "
        "one axial slice selected by reference whole-tumor area, native RAS display orientation, and colored reference/prediction overlays. "
        "Three case IDs and training seed 42 were fixed before the revised reserved-cohort evaluation. Dice captions are whole-volume scores.\n"
        "The supplied notebook's earlier validation subset included BRATS_143. Historical roles are documented in results/segmentation_audit/original_notebook_exposure.json; the fixed examples are current-model illustrations, not evidence of a historically untouched cohort.\n"
    )
    if synthetic:
        attribution = "# SYNTHETIC TEST FIXTURE\n\nThis preview uses artificial arrays and scores, without real MRI. The following attribution is the production-figure template.\n\n" + attribution
    (dest / "ATTRIBUTION.md").write_text(attribution)
