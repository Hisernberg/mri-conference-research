#!/usr/bin/env python3
"""Build a protected-evaluation script only after all main fits are verified."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from build_notebooks import materializer
from combine_segmentation_studies import combine, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
from locked_evaluate import MAIN_SOURCE_SHA, SEEDS
from qualitative_panels import POLICY_PATH, read_policy


def build(user="dasshovon", budget_seconds=20000):
    folders = [ROOT / f"results/segmentation_study_s{seed}" for seed in SEEDS]
    # Recheck all statistical and completion requirements before issuing a receipt.
    combined = ROOT / "results/segmentation_main"
    receipt = combine(folders, combined, SEEDS, [0], VARIANTS)
    snapshot = json.loads((ROOT / "audit/source_snapshots/segmentation_study_s42_v1.json").read_text())
    sources = snapshot["sources"]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if digest != MAIN_SOURCE_SHA:
        raise ValueError("The parent snapshot does not identify the frozen main study")
    file_hashes = {p: hashlib.sha256(text.encode()).hexdigest() for p, text in sources.items()}
    for rel, expected in file_hashes.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Inference source changed since main training: {rel}")
    parents = {}
    for seed, folder in zip(SEEDS, folders):
        state = json.loads((ROOT / f"runtime/study_s{seed}_collection.json").read_text())
        if not (state["phase"] == "ready_for_review" and state["status"] == "COMPLETE" and
                state["expected_source_sha256"] == MAIN_SOURCE_SHA):
            raise ValueError(f"Seed {seed} lacks a verified completed Kaggle source")
        snapshot_path = ROOT / f"audit/source_snapshots/segmentation_study_s{seed}_v{state['version']}.json"
        if not snapshot_path.exists():
            notebook_path = ROOT / f"runtime/study_s{seed}_bundle/02_segmentation_study.ipynb"
            notebook = json.loads(notebook_path.read_text())
            code_cells = ["".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                          for cell in notebook["cells"] if cell["cell_type"] == "code"]
            embedded = [ast.literal_eval(node.value) for code_cell in code_cells for node in ast.parse(code_cell).body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets)]
            if len(embedded) != 1 or embedded[0] != sources:
                raise ValueError("Queued notebook differs from the expected parent source")
            snapshot_path.write_text(json.dumps({"kernel": state["kernel"], "version": state["version"],
                "source_sha256": MAIN_SOURCE_SHA, "sources": embedded[0], "run_cell": code_cells[-1]},
                sort_keys=True, indent=2))
        inventory = json.loads((folder / "input_artifact_hashes.json").read_text())
        raw = ROOT / f"kaggle_outputs/segmentation_study_s{seed}/segmentation_results"
        for item in inventory:
            if hashlib.sha256((raw / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded parent artifact changed after verification: {seed}/{item['path']}")
        parents[str(seed)] = {"kernel": state["kernel"], "version": state["version"],
            "artifact_hashes": inventory,
            "verification_sha256": hashlib.sha256((folder / "verification.json").read_bytes()).hexdigest()}
    frozen_path = ROOT / "segmentation/configs/grouped_v2/locked_evaluation_protocol.json"
    gate = {"main_source_sha256": MAIN_SOURCE_SHA, "main_source_file_hashes": file_hashes,
        "combination_verification": receipt, "parents": parents,
        "frozen_protocol_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "qualitative_protocol": read_policy(json.loads(frozen_path.read_text())),
        "qualitative_protocol_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()}
    gate_path = ROOT / "audit/protected_development_gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True, indent=2))
    files = [ROOT / p for p in sources] + [ROOT / "segmentation/scripts/locked_evaluate.py",
        ROOT / "segmentation/scripts/qualitative_panels.py", POLICY_PATH, gate_path]
    code, source_hash = materializer(files)
    code += "\nimport subprocess\n"
    code += f"subprocess.run([sys.executable, str(SOURCE_ROOT / 'segmentation/scripts/locked_evaluate.py'), '--input', '/kaggle/input', '--work', str(WORK), '--gate', str(SOURCE_ROOT / 'audit/protected_development_gate.json'), '--budget-seconds', {str(budget_seconds)!r}], check=True)\n"
    compile(code, "protected_evaluation.py", "exec")
    dest = ROOT / "kaggle/protected_evaluation"; dest.mkdir(parents=True, exist_ok=True)
    (dest / "protected_evaluation.py").write_text(code)
    metadata = {"id": f"{user}/mri-segmentation-protected-evaluation", "title": "MRI Segmentation Protected Evaluation",
        "code_file": "protected_evaluation.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{user}/mri-segmentation-preparation-v2"] +
                          [f"{user}/mri-segmentation-study-s{seed}" for seed in SEEDS]}
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    result = {"source_sha256": source_hash, "parent_source_sha256": MAIN_SOURCE_SHA,
        "kernel": metadata["id"], "budget_seconds": budget_seconds,
        "frozen_protocol_sha256": gate["frozen_protocol_sha256"], "submission_status_at_build": "not_submitted"}
    (dest / "bundle_record.json").write_text(json.dumps(result, indent=2))
    (ROOT / "audit/source_snapshots/protected_evaluation_v1.json").write_text(json.dumps({
        "kernel": metadata["id"], "source_sha256": source_hash,
        "sources": {p.relative_to(ROOT).as_posix(): p.read_text() for p in sorted(files)},
        "budget_seconds": budget_seconds, "submission_status_at_build": "not_submitted"}, sort_keys=True, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--user", default="dasshovon")
    p.add_argument("--budget-seconds", type=int, default=20000)
    a = p.parse_args(); build(a.user, a.budget_seconds)
