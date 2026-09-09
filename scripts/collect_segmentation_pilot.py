#!/usr/bin/env python3
"""Collect private preparation/pilot evidence after the authorized watcher submits it.

This collector never submits or changes an experiment. A human-readable review
of the real pilot results is still required before choosing the extended budget.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle_api import connect

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/segmentation_collection.json"
ORCHESTRATION = ROOT / "runtime/segmentation_orchestration.json"
PREPARATION = "dasshovon/mri-segmentation-preparation-v2"
PILOT = "dasshovon/mri-segmentation-qmmf-v2"


def record(**fields):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps({"time_unix": time.time(), **fields}, indent=2))
    temp.replace(STATE)
    print(json.dumps(fields), flush=True)


def download(kernel, dest):
    subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_api.py"), "output", kernel,
                    "--out", str(dest), "--pattern", r"\.(json|csv|log)$"],
                   cwd=ROOT, check=True)


def main():
    while True:
        try:
            state = json.loads(ORCHESTRATION.read_text()) if ORCHESTRATION.exists() else {}
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if state.get("phase") == "pilot_submitted":
            response = state.get("response", {})
            if response.get("error"):
                raise RuntimeError("Pilot submission returned an error; inspect the orchestration record")
            break
        if state.get("preparation_status") in {"ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
            download(PREPARATION, ROOT / "kaggle_outputs/preparation")
            record(phase="preparation_failed_evidence_collected", status=state["preparation_status"])
            return
        record(phase="waiting_for_pilot_submission")
        time.sleep(30)
    download(PREPARATION, ROOT / "kaggle_outputs/preparation")
    api = connect()
    while True:
        status = api.kernels_status(PILOT).to_dict()["status"]
        record(phase="waiting_for_pilot_completion", kernel=PILOT, status=status)
        if status in {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
            break
        time.sleep(30)
    dest = ROOT / "kaggle_outputs/segmentation_pilot"
    download(PILOT, dest)
    if status != "COMPLETE":
        record(phase="pilot_failed_evidence_collected", kernel=PILOT, status=status)
        return
    sources = list(dest.rglob("session_status.json"))
    if len(sources) != 1:
        raise RuntimeError(f"Expected one session completion record; found {len(sources)}")
    subprocess.run([sys.executable, str(ROOT / "scripts/analyze_segmentation.py"),
                    str(sources[0].parent), "--output", str(ROOT / "results/segmentation_pilot")],
                   cwd=ROOT, check=True)
    record(phase="pilot_ready_for_review", kernel=PILOT, status=status,
           result_dir="results/segmentation_pilot")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record(phase="collector_error", error_type=type(exc).__name__, message=str(exc))
        raise
