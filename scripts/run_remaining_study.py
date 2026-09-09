#!/usr/bin/env python3
"""Submit the already frozen third seed after a verified run frees a GPU slot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from kaggle_api import connect, native, quota_summary

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/remaining_study_queue.json"
SOURCE_SHA = "037bc0a99fab8f294bc710274b67b96b190ef4c4dbab6303fa64238040ec4d0a"


def record(**fields):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps({"time_unix": time.time(), **fields}, indent=2))
    temp.replace(STATE)
    print(json.dumps(fields), flush=True)


def submission_reserve(remaining_active_seconds, already_reserved_seconds):
    # remaining_unreserved has already deducted Kaggle's running-job reservation.
    # Add only the part of our conservative active-job estimate not covered by it.
    unreserved_active = max(0., remaining_active_seconds - already_reserved_seconds)
    return 30000 + 14400 + unreserved_active + 300


def main():
    if STATE.exists() and json.loads(STATE.read_text()).get("phase") in {
            "submitting", "submitted", "submission_uncertain", "complete"}:
        raise RuntimeError("This queue has already attempted submission; reconcile its recorded kernel before restarting")
    bundle = ROOT / "runtime/study_s44_bundle"
    hashes = json.loads((bundle / "bundle_hashes.json").read_text())
    for filename, expected in hashes.items():
        if hashlib.sha256((bundle / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen queued bundle was modified")
    api = connect()
    while True:
        states, completed = {}, []
        for seed in [42, 43]:
            path = ROOT / f"runtime/study_s{seed}_collection.json"
            states[seed] = json.loads(path.read_text()) if path.exists() else {}
            verification = ROOT / f"results/segmentation_study_s{seed}/verification.json"
            if verification.exists() and states[seed].get("phase") == "ready_for_review":
                checked = json.loads(verification.read_text())
                if not checked["all_planned_completed"]:
                    record(phase="needs_run_repair", seed=seed, reason="partial matrix; third seed not submitted")
                    return
                completed.append(seed)
            if states[seed].get("phase") in {"failed_run_evidence_collected", "analysis_failed", "source_verification_failed"}:
                record(phase="needs_run_repair", seed=seed, reason=states[seed]["phase"])
                return
        if completed:
            quota = quota_summary(api.quota_view())
            other_allowance = 0.
            for seed in set([42, 43]) - set(completed):
                log = ROOT / f"runtime/study_s{seed}_collection.log"
                first = json.loads(log.read_text().splitlines()[0])
                other_allowance += max(0., 30000 - (time.time() - first["time_unix"]))
            required = submission_reserve(other_allowance, quota["gpu"]["time_reserved"])
            if quota["gpu"]["remaining_unreserved"] < required:
                record(phase="quota_reserve_insufficient", available=quota["gpu"]["remaining_unreserved"],
                       required=required, completed_seeds=completed)
                return
            record(phase="submitting", seed=44, completed_seeds=completed,
                   remaining_quota_seconds=quota["gpu"]["remaining_unreserved"], required_reserve_seconds=required)
            try:
                response = native(api.kernels_push(str(bundle), acc="NvidiaTeslaT4", timeout="30000"))
            except Exception as exc:
                record(phase="submission_uncertain", seed=44, error_type=type(exc).__name__)
                raise
            if response.get("error") or not response.get("versionNumber"):
                record(phase="submission_uncertain", seed=44, response=response)
                raise RuntimeError("Reconcile the third-seed submission before retrying")
            record(phase="submitted", seed=44, response=response, source_sha256=SOURCE_SHA)
            version = response["versionNumber"]
            break
        record(phase="waiting_for_verified_gpu_slot", completed_seeds=completed)
        time.sleep(30)
    # Keep this process alive while its collector runs, so no detached job is lost.
    with (ROOT / "runtime/study_s44_collection.log").open("a") as log:
        process = subprocess.run([sys.executable, str(ROOT / "scripts/collect_kernel.py"),
            "dasshovon/mri-segmentation-study-s44", "--version", str(version), "--expected-source", SOURCE_SHA,
            "--output", str(ROOT / "kaggle_outputs/segmentation_study_s44"),
            "--state", str(ROOT / "runtime/study_s44_collection.json"),
            "--analysis-output", str(ROOT / "results/segmentation_study_s44")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        record(phase="collector_failed", seed=44, returncode=process.returncode)
        return
    final_state = json.loads((ROOT / "runtime/study_s44_collection.json").read_text())
    verification = ROOT / "results/segmentation_study_s44/verification.json"
    checked = json.loads(verification.read_text()) if verification.exists() else {}
    if not checked.get("all_planned_completed"):
        record(phase="needs_run_repair", seed=44,
               reason="third-seed collector did not verify the complete matrix",
               collector_phase=final_state.get("phase"))
        return
    record(phase="complete", seed=44, source_sha256=SOURCE_SHA)


if __name__ == "__main__":
    main()
