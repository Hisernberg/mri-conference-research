#!/usr/bin/env python3
"""Wait for all verified main fits, then execute the prespecified protected study.

No metric ranking determines model inclusion. Submission is recorded before the
API call; an uncertain response must be reconciled instead of blindly retried.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from build_locked_bundle import build
from kaggle_api import connect, native, quota_summary

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/protected_queue.json"


def record(**fields):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps({"time_unix": time.time(), **fields}, indent=2))
    temp.replace(STATE)
    print(json.dumps(fields), flush=True)


def main():
    with (ROOT / "runtime/protected_queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists() and json.loads(STATE.read_text()).get("phase") in {
                "submitting", "submitted", "submission_uncertain", "complete", "collecting", "analyzing"}:
            raise RuntimeError("Protected submission already attempted; reconcile the recorded run before restarting")
        while True:
            ready = []
            for seed in [42, 43, 44]:
                path = ROOT / f"runtime/study_s{seed}_collection.json"
                state = json.loads(path.read_text()) if path.exists() else {}
                if state.get("phase") in {"failed_run_evidence_collected", "analysis_failed", "source_verification_failed", "missing_session_ledger"}:
                    record(phase="needs_development_repair", seed=seed, reason=state["phase"])
                    return
                if state.get("phase") == "ready_for_review":
                    checked = json.loads((ROOT / f"results/segmentation_study_s{seed}/verification.json").read_text())
                    if not checked["all_planned_completed"]:
                        record(phase="needs_development_repair", seed=seed, reason="incomplete main matrix")
                        return
                    ready.append(seed)
            if len(ready) == 3:
                break
            record(phase="waiting_for_complete_development", verified_seeds=ready,
                   locked_test_opened=False)
            time.sleep(30)
        bundle = build(budget_seconds=20000)
        api = connect(); quota = quota_summary(api.quota_view())
        if quota["gpu"]["remaining_unreserved"] < 21300:
            record(phase="quota_reserve_insufficient", required_seconds=21300,
                   available_seconds=quota["gpu"]["remaining_unreserved"], locked_test_opened=False,
                   bundle=bundle)
            return
        record(phase="submitting", bundle=bundle, remaining_quota_seconds=quota["gpu"]["remaining_unreserved"])
        try:
            response = native(api.kernels_push(str(ROOT / "kaggle/protected_evaluation"),
                                              acc="NvidiaTeslaT4", timeout="21000"))
        except Exception as exc:
            record(phase="submission_uncertain", bundle=bundle, error_type=type(exc).__name__)
            raise
        if response.get("error") or not response.get("versionNumber"):
            record(phase="submission_uncertain", bundle=bundle, response=response)
            raise RuntimeError("Reconcile protected-evaluation submission before retrying")
        record(phase="submitted", bundle=bundle, response=response)
        with (ROOT / "runtime/protected_collection.log").open("a") as log:
            process = subprocess.run([sys.executable, str(ROOT / "scripts/collect_kernel.py"), bundle["kernel"],
                "--version", str(response["versionNumber"]), "--expected-source", bundle["source_sha256"],
                "--output", str(ROOT / "kaggle_outputs/protected_evaluation"),
                "--state", str(ROOT / "runtime/protected_collection.json"),
                "--pattern", r"\.(json|csv|log|png)$"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads((ROOT / "runtime/protected_collection.json").read_text())
        if process.returncode or state.get("phase") != "ready_for_review":
            record(phase="needs_protected_repair", bundle=bundle, collector_state=state,
                   returncode=process.returncode)
            return
        record(phase="analyzing", bundle=bundle, response=response)
        process = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_locked_segmentation.py"),
            str(ROOT / "kaggle_outputs/protected_evaluation/protected_results"),
            "--output", str(ROOT / "results/segmentation_protected")], cwd=ROOT)
        checked_path = ROOT / "results/segmentation_protected/verification.json"
        checked = json.loads(checked_path.read_text()) if checked_path.exists() else {}
        qualitative = checked.get("qualitative", {})
        if (process.returncode or not checked.get("all_planned_completed") or
                not qualitative.get("verified") or qualitative.get("synthetic_fixture", True)):
            record(phase="needs_protected_repair", bundle=bundle,
                   reason="independent protected verification failed", returncode=process.returncode)
            return
        record(phase="complete", bundle=bundle, response=response,
               completed_checkpoint_evaluations=checked["completed_checkpoint_evaluations"])


if __name__ == "__main__":
    main()
