#!/usr/bin/env python3
"""Audit recorded historical case use; optionally freeze a metadata-only sensitivity."""
import argparse
import ast
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/src"))
from qmmf.utils import sha256_obj

MAIN_SHA = "037bc0a99fab8f294bc710274b67b96b190ef4c4dbab6303fa64238040ec4d0a"
EVALUATE_SHA = "9f82445007dd6c7bb22fada8e9241c0e991fa4ea19bc579e251d7f67c7778188"
WORKER_SHA = "82b9736fe092a004cb466a932b173ccf76b5a603cea928b8f1527e2c127f1ac8"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_split(path):
    value = read(path)
    content = {k: v for k, v in value.items() if k != "split_hash"}
    require(sha256_obj(content)[:16] == value["split_hash"], "Ordered split digest differs")
    return value


def derive_exposure(old, new, validation_limit, trained_folds=(0,)):
    """Use complete groups, never score-dependent or case-only exclusions."""
    mapping = new["case_to_group"]
    all_cases = set(old["development"]) | set(old["locked_test"])
    test = set(new["locked_test"])
    require(all_cases == set(mapping), "Case universes differ")
    require(set(new["development"]) | test == all_cases, "New partition is incomplete")
    require(not (set(new["development"]) & test), "New development/test case overlap")
    test_groups = {mapping[c] for c in test}
    require(not (test_groups & {mapping[c] for c in new["development"]}),
            "New development/test group overlap")
    require(validation_limit > 0, "Validation limit must be positive")
    original_train, original_validation = set(), set()
    for fold in trained_folds:
        row = old["folds"][str(fold)]
        original_train.update(row["train"])
        original_validation.update(row["inner_val"][:validation_limit])
    normalizer_pool = set().union(*(set(row["train"]) for row in old["folds"].values()))
    old_development = set(old["development"])
    roles = {case: role for role, cases in old["folds"]["0"].items() for case in cases}
    roles.update({case: "locked_test" for case in old["locked_test"]})
    require(set(roles) == all_cases, "Original fold-0 roles are incomplete")
    pools = {
        "original_active_fold_training": original_train,
        "original_validation_subset": original_validation,
        "original_all_fold_normalizer_pool": normalizer_pool,
        "original_development": old_development,
    }
    group_pools = {name: {mapping[c] for c in cases} for name, cases in pools.items()}
    eligible = test_groups - group_pools["original_development"]
    rows = []
    for case in sorted(test):
        group = mapping[case]
        row = {"case_id": case, "group_id": group, "original_fold0_role": roles[case]}
        for name, cases in pools.items():
            row[name + "_direct"] = case in cases
            row[name + "_group"] = group in group_pools[name]
        row["old_locked_only_sensitivity"] = group in eligible
        rows.append(row)
    summary = {}
    for name, cases in pools.items():
        direct = test & cases
        overlap = test_groups & group_pools[name]
        summary[name] = {
            "historical_case_count": len(cases),
            "direct_reserved_case_overlap": len(direct),
            "reserved_group_overlap": len(overlap),
            "reserved_cases_in_overlapping_groups": sum(mapping[c] in overlap for c in test),
        }
    return {
        "rows": rows, "overlap": summary,
        "historical_validation_case_ids": sorted(original_validation),
        "sensitivity_group_ids": sorted(eligible),
        "sensitivity_full_case_ids": sorted(c for c in test if mapping[c] in eligible),
    }


def original_source_evidence(root):
    source_path = root / "audit/buet_source.py"
    output_path = root / "audit/buet_outputs.txt"
    input_receipt = read(root / "audit/original_hashes.json")
    entry = next(r for r in input_receipt["files"] if r["file"] == "buet-1.ipynb")
    require(digest(source_path) == entry["source_snapshot_sha256"], "Original source snapshot differs")
    require(digest(output_path) == entry["text_output_snapshot_sha256"], "Original output snapshot differs")
    tree = ast.parse(source_path.read_text())

    def assignment(name):
        return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))

    scope = ast.literal_eval(assignment("SCOPE"))
    presets = assignment("PRESETS")
    preset = next(value for key, value in zip(presets.keys, presets.values)
                  if ast.literal_eval(key) == scope)
    require(isinstance(preset, ast.Call) and isinstance(preset.func, ast.Name)
            and preset.func.id == "dict" and not preset.args
            and all(k.arg is not None for k in preset.keywords), "Historical preset syntax differs")
    fields = {item.arg: item.value for item in preset.keywords}
    limit = ast.literal_eval(fields["max_val_cases"])
    folds = ast.literal_eval(fields["folds"])
    require(scope == "pilot" and limit == 16 and folds == [0], "Historical execution scope differs")
    payload = ast.literal_eval(assignment("_PAYLOAD_B64"))
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode("".join(payload.split()))), mode="r:gz") as tf:
        # Read members as data; never execute or extract the embedded repository.
        evaluation = tf.extractfile("src/qmmf/evaluate.py").read()
        worker = tf.extractfile("scripts/run_worker.py").read()
    require(hashlib.sha256(evaluation).hexdigest() == EVALUATE_SHA, "Historical evaluation source differs")
    require(hashlib.sha256(worker).hexdigest() == WORKER_SHA, "Historical worker source differs")
    require("ids = list(case_ids)[:max_cases] if max_cases else list(case_ids)" in evaluation.decode(),
            "Historical validation selection rule differs")
    output = output_path.read_text()
    split_hash = re.search(r"split hash\s+([0-9a-f]{16})", output).group(1)
    manifest_hash = re.search(r"manifest hash\s+([0-9a-f]{16})", output).group(1)
    require("quality normalisers fitted for folds: ['0', '1', '2', '3']" in output,
            "Four historical normalizer records are not logged")
    require(bool(re.search(r"epoch\s+\d+\s+loss\s+[0-9.eE+-]+\s+score", output)),
            "Historical validation progress is not logged")
    return {
        "original_notebook_sha256_recorded_later": entry["sha256"],
        "original_source_sha256": digest(source_path),
        "original_text_outputs_sha256": digest(output_path),
        "original_evaluate_member_sha256": EVALUATE_SHA,
        "original_worker_member_sha256": WORKER_SHA,
        "logged_split_hash": split_hash, "logged_manifest_hash": manifest_hash,
        "scope": scope, "active_training_folds": folds, "validation_case_limit": limit,
        "validation_selection_rule": "first max_val_cases entries in the ordered inner_val list",
        "normalizer_folds_prepared_or_loaded": [0, 1, 2, 3],
    }


def audit(root=ROOT, freeze=False):
    root = Path(root)
    evidence = original_source_evidence(root)
    old = ordered_split(root / "results/segmentation_audit/splits.json")
    new = ordered_split(root / "segmentation/configs/grouped_v2/splits.json")
    require(old["split_hash"] == evidence["logged_split_hash"], "Logged historical split differs")
    fingerprint = read(root / "results/segmentation_audit/dataset_fingerprint.json")
    require(fingerprint["manifest_hash"] == evidence["logged_manifest_hash"], "Historical manifest differs")
    data = derive_exposure(old, new, evidence["validation_case_limit"], evidence["active_training_folds"])
    require(data["overlap"]["original_active_fold_training"]["reserved_group_overlap"] == 0,
            "Original active training groups overlap the revised reserved cohort")
    base_path = root / "segmentation/configs/grouped_v2/locked_evaluation_protocol.json"
    base = read(base_path)
    require(set(base["full_modality_case_ids"]) == set(new["locked_test"]), "Reserved cases differ")
    qualitative = read(root / "audit/qualitative_protocol.json")
    by_case = {row["case_id"]: row for row in data["rows"]}
    qual_history = [by_case[case] for case in qualitative["case_ids"]]
    target = root / "results/segmentation_audit/original_notebook_exposure.json"
    old_record = read(target) if target.exists() else {}
    recorded = old_record.get("recorded_utc", datetime.now(timezone.utc).isoformat())
    report = {
        "recorded_utc": recorded, "source_evidence": evidence,
        "current_split_hash": new["split_hash"], "current_reserved_cases": len(new["locked_test"]),
        "current_reserved_groups": len(set(new["case_to_group"][c] for c in new["locked_test"])),
        "overlap": data["overlap"],
        "historical_validation_case_ids": data["historical_validation_case_ids"],
        "sensitivity_group_ids": data["sensitivity_group_ids"],
        "sensitivity_full_case_ids": data["sensitivity_full_case_ids"],
        "qualitative_case_history": qual_history,
        "interpretation": [
            "Protected means reserved from fitting and selection in the revised experiment.",
            "The supplied original notebook contains earlier validation involving some reserved groups.",
            "Only fold 0 trained original models; the other fold normalizers were prepared or loaded but their model fits are not recorded.",
            "Revised fits do not reuse the original weights or normalizers; this is historical development exposure, not demonstrated leakage within the revised fits.",
            "Old-locked-only groups are defined from recorded partitions; undocumented earlier use and patient identity remain unknown.",
        ],
        "raw_images_or_revised_reserved_performance_read": False,
    }
    if old_record:
        require(report == old_record, "Historical audit changed; retain and reconcile the earlier record")
    else:
        target.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = target.with_suffix(".csv")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(data["rows"][0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(data["rows"])
    csv_payload = buffer.getvalue().encode()
    if csv_path.exists():
        require(csv_path.read_bytes() == csv_payload, "Historical case table changed")
    else:
        csv_path.write_bytes(csv_payload)
    protocol_path = root / "audit/historical_exposure_sensitivity_protocol.json"
    if freeze:
        groups = set(data["sensitivity_group_ids"])
        subset_cases = sorted(c for c in base["robustness_case_ids"] if new["case_to_group"][c] in groups)
        require(len(groups) >= 2 and len(subset_cases) == len(groups), "Invalid sensitivity cohort")
        frozen = read(protocol_path) if protocol_path.exists() else {}
        if not frozen:
            state = read(root / "runtime/protected_queue.json")
            require(state["phase"] == "waiting_for_complete_development" and not state["locked_test_opened"],
                    "Cannot freeze this sensitivity after revised reserved-cohort execution started")
        protocol = {
            "recorded_utc": frozen.get("recorded_utc", datetime.now(timezone.utc).isoformat()),
            "analysis_class": "exploratory historical-cohort sensitivity, added after the initial plan",
            "timing": "First two main development seeds had been reviewed; no revised reserved-cohort result had been produced or reviewed.",
            "original_split_hash": old["split_hash"], "current_split_hash": new["split_hash"],
            "main_source_sha256": MAIN_SHA,
            "base_quantitative_protocol_sha256": digest(base_path),
            "historical_audit_sha256": digest(target), "historical_case_table_sha256": digest(csv_path),
            "cohort_rule": "Retain complete revised reserved groups only if no member belonged to the original development partition; all members must be in the original locked partition.",
            "group_ids": data["sensitivity_group_ids"],
            "full_case_ids": data["sensitivity_full_case_ids"], "robustness_case_ids": subset_cases,
            "models": base["models"], "seeds": base["seeds"],
            "metrics": ["full_group_macro_dice", "mean_subset_macro_dice", "worst_subset_macro_dice"],
            "bootstrap_draws": 10000, "bootstrap_seed": 20260909,
            "statistics": "Same equal-group estimands and paired group bootstrap as the full reserved analysis; retain all three seeds per draw, report seed SD and unadjusted plus Bonferroni intervals for three controls per endpoint.",
            "full_cohort_report_retained": True, "new_training_or_inference": False,
            "qualitative_examples_changed": False,
            "limitations": "Small, selected internal subset; no patient-identity guarantee, external validation, or proof against undocumented historical use.",
        }
        if frozen:
            require(protocol == frozen, "Frozen historical sensitivity changed")
        else:
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "overlap": report["overlap"],
        "sensitivity_groups": len(data["sensitivity_group_ids"]),
        "sensitivity_cases": len(data["sensitivity_full_case_ids"]),
        "protocol_frozen": protocol_path.exists(),
        "protocol_sha256": digest(protocol_path) if protocol_path.exists() else None,
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-sensitivity", action="store_true")
    args = parser.parse_args()
    audit(freeze=args.freeze_sensitivity)
