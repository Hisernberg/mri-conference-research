#!/usr/bin/env python3
"""Evaluate the frozen protected cohort using existing inner-selected checkpoints.

This supporting script cannot train or fit a normalizer. A locally verified
27-fit development receipt and unchanged parent artifacts are required first.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from conference_run import context, find_prepared, save
from qmmf.config import ExperimentConfig, MODALITIES
from qmmf.evaluate import evaluate_cases, evaluate_all_subsets
from qmmf.grouping import group_means
from qmmf.models import build_model
from qmmf.quality import QualityNormalizer
from qmmf.shapley import shapley_from_subset_table
from qmmf.splits import Splits, validate_splits
from qmmf.subsets import ALL_SUBSETS, ALL_SUBSET_KEYS, subset_key
from qmmf.utils import environment_fingerprint, seed_everything
from qualitative_panels import POLICY_PATH, read_policy, save_model_panels

MAIN_SOURCE_SHA = "037bc0a99fab8f294bc710274b67b96b190ef4c4dbab6303fa64238040ec4d0a"
MODELS = ["qmmf", "hemis", "no_quality", "unet25d"]
SEEDS = [42, 43, 44]
FROZEN = Path(__file__).resolve().parents[1] / "configs/grouped_v2"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol, splits):
    validate_splits(splits)
    require(protocol["split_hash"] == splits.split_hash, "Protected split hash mismatch")
    require(protocol["models"] == MODELS and protocol["seeds"] == SEEDS,
            "Protected model/seed matrix changed")
    require(protocol["training_fold"] == 0 and protocol["threshold"] == .5,
            "Protected fold or threshold changed")
    full, robust = protocol["full_modality_case_ids"], protocol["robustness_case_ids"]
    require(len(full) == len(set(full)) and set(full) == set(splits.locked_test),
            "Protected full cohort differs from frozen test")
    require(len(robust) == len(set(robust)) and set(robust) <= set(full),
            "Invalid protected representative cases")
    mapping = splits.case_to_group
    groups = {mapping[c] for c in full}
    require(len(groups) == protocol["n_full_modality_groups"] == 51,
            "Protected group count changed")
    require(len(robust) == protocol["n_robustness_groups"] == len(groups),
            "Protected robustness group count changed")
    expected = {min([c for c in full if mapping[c] == group],
                    key=lambda c: hashlib.sha256(f"20260909|locked|{c}".encode()).hexdigest())
                for group in groups}
    require(set(robust) == expected, "Protected representatives violate frozen hash rule")
    development = {mapping[c] for role in splits.folds["0"].values() for c in role}
    require(not development & groups, "Protected group overlaps development")


def validate_checkpoint(payload, cfg, completion, normalizer, splits):
    require(payload["config_hash"] == completion["config_hash"] == cfg.config_hash(),
            "Selected checkpoint configuration hash differs")
    require(ExperimentConfig.from_dict(payload["config"]).to_dict() == cfg.to_dict(),
            "Selected checkpoint configuration differs")
    state = payload["state"]
    require(state["best_epoch"] == completion["best_epoch"] and
            state["epoch"] == completion["best_epoch"] + 1,
            "Checkpoint is not the recorded inner-selected epoch")
    require(np.isclose(state["best_score"], completion["inner_selection_score"], rtol=1e-8),
            "Checkpoint inner selection score differs")
    train = set(splits.folds[str(cfg.fold)]["train"])
    require(set(normalizer["training_case_ids"]) == train,
            "Checkpoint normalizer was not fitted on the training cohort")
    require(set(normalizer["training_group_ids"]) == {splits.case_to_group[c] for c in train},
            "Checkpoint normalizer training groups differ")
    require(payload.get("quality_normalizer") == normalizer["normalizer"],
            "Checkpoint normalizer differs from training provenance")
    require(all(torch.isfinite(value).all().item() for value in payload["model"].values()),
            "Checkpoint has non-finite model tensors")


def validate_parents(input_dir, gate, protocol, splits):
    require(gate["main_source_sha256"] == MAIN_SOURCE_SHA,
            "Unrecognized parent training source")
    receipt = gate["combination_verification"]
    require(receipt["all_required_completed"] and receipt["completed_model_fits"] == 27 and
            receipt["seeds"] == SEEDS and receipt["folds"] == [0] and
            receipt["split_hash"] == splits.split_hash and not receipt["locked_test_opened"],
            "All 27 development fits must be independently verified first")
    require(gate["frozen_protocol_sha256"] == sha(FROZEN / "locked_evaluation_protocol.json"),
            "Protected protocol changed after the development gate")
    qualitative = read_policy(protocol)
    require(gate["qualitative_protocol"] == qualitative and
            gate["qualitative_protocol_sha256"] == sha(POLICY_PATH),
            "Prespecified qualitative protocol differs from the development gate")
    require(qualitative["quantitative_protocol_sha256"] == gate["frozen_protocol_sha256"],
            "Qualitative panels are attached to a different quantitative protocol")
    # Verify the inference library itself against the submitted main snapshot.
    source_root = Path(__file__).resolve().parents[2]
    for rel, expected in gate["main_source_file_hashes"].items():
        require(sha(source_root / rel) == expected, f"Main source file changed: {rel}")
    indexed = {}
    for path in Path(input_dir).rglob("protocol_lock.json"):
        if path.parent.name != "segmentation_results":
            continue
        parent = path.parent
        parent_protocol = read(path)
        seed = parent_protocol["seed"]
        require(seed in SEEDS and seed not in indexed, "Repeated or unexpected parent seed")
        expected = gate["parents"][str(seed)]
        for item in expected["artifact_hashes"]:
            require(sha(parent / item["path"]) == item["sha256"],
                    f"Verified parent artifact changed: seed {seed}/{item['path']}")
        sources = {rel: (parent.parent / "mri_source" / rel).read_text()
                   for rel in gate["main_source_file_hashes"]}
        digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        require(digest == MAIN_SOURCE_SHA, "Attached parent source snapshot differs")
        status = read(parent / "session_status.json")
        require(parent_protocol["scope"] == "study" and parent_protocol["fold"] == 0 and
                set(status["completed_variants"]) == set(receipt["variants"]) and
                not status["locked_test_opened"], "Parent study is incomplete or test-exposed")
        require(set(parent_protocol["names"]) == set(receipt["variants"]), "Parent matrix differs")
        configs = parent_protocol["configs"]
        selected = {}
        for item in (parent / "qmmf_runs").rglob("completed.json"):
            completion = read(item)
            name = completion["name"]
            if name not in MODELS:
                continue
            require(name not in selected, "Repeated selected parent model")
            cfg = ExperimentConfig.from_dict(configs[name])
            require(cfg.seed == cfg.train.seed == seed and cfg.fold == 0 and
                    cfg.train.max_epochs == 120 and cfg.train.steps_per_epoch == 100 and
                    cfg.train.val_every == 10, "Parent schedule differs from frozen main study")
            require((item.parent / "best.pt").is_file(), "Selected checkpoint missing")
            selected[name] = str(item.parent)
        require(set(selected) == set(MODELS), "A protected model checkpoint is missing")
        indexed[seed] = {"root": str(parent), "selected": selected}
    require(set(indexed) == set(SEEDS), "Attach all three immutable main-seed outputs")
    return indexed


def worker(prepared, parent, run, name, seed, dest):
    torch.set_num_threads(2); seed_everything(seed)
    protocol = read(FROZEN / "locked_evaluation_protocol.json")
    _, cache, splits, _, spacing = context(Path(prepared), fold=0, phase="study")
    validate_protocol(protocol, splits)
    parent, run, dest = Path(parent), Path(run), Path(dest)
    cfg = ExperimentConfig.from_dict(read(parent / "protocol_lock.json")["configs"][name])
    completion, normalizer = read(run / "completed.json"), read(run / "quality_normalizer.json")
    require(completion["seed"] == cfg.seed == seed and completion["name"] == name,
            "Wrong protected checkpoint identity")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, cfg, completion, normalizer, splits)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha(run / "best.pt")
    provenance = {"seed": seed, "model": name, "parent_source_sha256": MAIN_SOURCE_SHA,
        "checkpoint_sha256": checkpoint_hash, "parent_config_hash": cfg.config_hash(),
        "selected_epoch": completion["best_epoch"], "inner_selection_score": completion["inner_selection_score"],
        "normalizer": normalizer, "parent_completion": completion,
        "config": cfg.to_dict(), "training_performed": False,
        "normalizer_fitted": False, "frozen_protocol_sha256": sha(FROZEN / "locked_evaluation_protocol.json")}
    if (dest / "checkpoint_provenance.json").exists():
        require(read(dest / "checkpoint_provenance.json") == provenance,
                "Cannot resume protected evaluation with a different checkpoint")
    save(dest / "checkpoint_provenance.json", provenance)
    if (dest / "completed.json").exists():
        return
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "Protected evaluation requires the configured GPU")
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.quality_normalizer = QualityNormalizer.from_dict(checkpoint["quality_normalizer"])
    del checkpoint
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    prefix = f"protected {name} seed {seed}"
    if not (dest / "full_summary.json").exists():
        result = evaluate_cases(model, protocol["full_modality_case_ids"], cache, cfg, spacing,
            threshold=protocol["threshold"], device=device, detailed_metrics=False, progress=prefix)
        pd.DataFrame([r.__dict__ for r in result["rows"]]).to_csv(dest / "full_case_metrics.csv", index=False)
        grouped = group_means(result["per_case_macro"], splits.case_to_group)
        save(dest / "full_summary.json", {"summary": result["summary"],
            "per_case_macro": result["per_case_macro"], "per_group_macro": grouped,
            "primary_group_macro_dice": float(np.mean(list(grouped.values())))})
    # Persist each subset so an interrupted session can continue without reselection.
    pieces = []
    for index, subset in enumerate(ALL_SUBSETS, 1):
        key = subset_key(subset); path = dest / "subset_progress" / f"{key}.json"
        if not path.exists():
            result = evaluate_all_subsets(model, protocol["robustness_case_ids"], cache, cfg, spacing,
                threshold=protocol["threshold"], device=device, subsets=[subset])
            save(path, result)
        pieces.append(read(path))
        print(f"[{prefix}] subset {index}/15 saved; {time.monotonic()-start:.1f}s this attempt", flush=True)
    per_case = {cid: {} for cid in protocol["robustness_case_ids"]}
    regions = {cid: {} for cid in per_case}
    for piece in pieces:
        for cid in per_case:
            per_case[cid].update(piece["per_case"][cid])
            for region, scores in piece["per_case_region"][cid].items():
                regions[cid].setdefault(region, {}).update(scores)
    require(all(set(table) == set(ALL_SUBSET_KEYS) for table in per_case.values()), "Incomplete subset table")
    subsets = {"per_case": per_case, "per_case_region": regions,
        "mean_over_subsets": {c: float(np.mean(list(v.values()))) for c, v in per_case.items()},
        "worst_subset": {c: float(np.min(list(v.values()))) for c, v in per_case.items()},
        "shapley": {c: shapley_from_subset_table(v) for c, v in per_case.items()}}
    save(dest / "modality_subsets.json", subsets)
    summary = read(dest / "full_summary.json")
    qualitative_start = time.monotonic()
    panels = save_model_panels(model, cache, cfg, spacing, protocol, summary, name, seed,
                              checkpoint_hash, splits, dest, device)
    qualitative_seconds = time.monotonic() - qualitative_start
    save(dest / "completed.json", {"completed": True, "name": name, "seed": seed,
        "scope": "protected_image_groups", "locked_test_opened": True,
        "checkpoint_sha256": checkpoint_hash, "parent_config_hash": cfg.config_hash(),
        "full_group_macro_dice": summary["primary_group_macro_dice"],
        "mean_subset_macro_dice": float(np.mean(list(subsets["mean_over_subsets"].values()))),
        "worst_subset_macro_dice": float(np.mean(list(subsets["worst_subset"].values()))),
        "full_cases": len(protocol["full_modality_case_ids"]), "full_groups": 51,
        "robustness_groups": 51, "subsets_per_group": 15,
        "qualitative_cases": len(panels["panels"]) if panels else 0,
        "qualitative_wall_seconds": qualitative_seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated()/2**30,
        "attempt_wall_seconds": time.monotonic()-start})


def evaluate(input_dir, work, gate_path, budget_seconds):
    torch.set_num_threads(2)
    gate, protocol = read(gate_path), read(FROZEN / "locked_evaluation_protocol.json")
    splits = Splits.load(FROZEN / "splits.json")
    validate_protocol(protocol, splits)
    parents = validate_parents(input_dir, gate, protocol, splits)
    prepared = find_prepared(input_dir)
    context(prepared, fold=0, phase="study")  # metadata/geometry only; no volume reads
    require(torch.cuda.device_count() > 0, "Protected evaluation requires GPU")
    dest = Path(work) / "protected_results"; dest.mkdir(parents=True, exist_ok=True)
    save(dest / "development_gate.json", gate)
    save(dest / "frozen_protocol.json", protocol)
    save(dest / "qualitative_protocol.json", read_policy(protocol))
    save(dest / "evaluation_protocol.json", {**protocol, "locked_test_opened": True,
        "main_source_sha256": MAIN_SOURCE_SHA, "training_performed": False,
        "normalizer_fitted": False, "execution_budget_seconds": budget_seconds})
    splits.save(dest / "splits.json")
    save(dest / "environment.json", environment_fingerprint())
    save(dest / "access_record.json", {"time_unix": time.time(), "locked_test_opened": True,
        "development_gate_sha256": sha(gate_path), "frozen_protocol_sha256": sha(FROZEN / "locked_evaluation_protocol.json")})
    start = time.monotonic()
    pending = [(seed, name) for seed in SEEDS for name in MODELS]
    active, failures = {}, {}
    while pending or active:
        for gpu in range(min(torch.cuda.device_count(), 2)):
            if gpu in active or not pending or time.monotonic()-start >= budget_seconds:
                continue
            seed, name = pending.pop(0); parent = parents[seed]
            out = dest / "runs" / f"{name}_s{seed}"
            cmd = [sys.executable, str(Path(__file__)), "--worker", "--prepared", str(prepared),
                "--parent", parent["root"], "--run", parent["selected"][name],
                "--name", name, "--seed", str(seed), "--work", str(out)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            active[gpu] = (seed, name, subprocess.Popen(cmd, env=env))
            print("START PROTECTED", seed, name, "GPU", gpu, flush=True)
        for gpu, (seed, name, process) in list(active.items()):
            code = process.poll()
            if code is not None:
                if code:
                    failures[f"{name}_s{seed}"] = code
                del active[gpu]
        if time.monotonic()-start >= budget_seconds:
            for seed, name, process in active.values():
                process.terminate(); process.wait(timeout=30)
                failures[f"{name}_s{seed}"] = "budget_exhausted_partial_subsets_saved"
            active.clear(); break
        time.sleep(5)
    completed = [read(p) for p in (dest / "runs").rglob("completed.json")]
    save(dest / "session_status.json", {"planned": [{"seed": s, "name": n} for s in SEEDS for n in MODELS],
        "completed": [{"seed": r["seed"], "name": r["name"]} for r in completed],
        "all_planned_completed": len(completed) == 12, "pending": pending, "failures": failures,
        "locked_test_opened": True, "wall_seconds": time.monotonic()-start})
    print("PROTECTED SESSION COMPLETE", len(completed), "/12", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--gate", type=Path); p.add_argument("--budget-seconds", type=int, default=20000)
    p.add_argument("--worker", action="store_true"); p.add_argument("--prepared", type=Path)
    p.add_argument("--parent", type=Path); p.add_argument("--run", type=Path)
    p.add_argument("--name", choices=MODELS); p.add_argument("--seed", type=int, choices=SEEDS)
    a = p.parse_args()
    if a.worker:
        worker(a.prepared, a.parent, a.run, a.name, a.seed, a.work)
    else:
        evaluate(a.input, a.work, a.gate, a.budget_seconds)
