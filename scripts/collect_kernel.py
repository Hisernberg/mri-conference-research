#!/usr/bin/env python3
"""Monitor one submitted private kernel and collect tables/logs; never launches jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from kaggle_api import connect

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--expected-source", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--analysis-output", type=Path)
    p.add_argument("--pattern", default=r"\.(json|csv|log)$",
                   help="Filename pattern for this run's declared output artifacts")
    a = p.parse_args()
    a.state.parent.mkdir(parents=True, exist_ok=True)
    def record(**fields):
        data = {"time_unix": time.time(), "kernel": a.kernel, "version": a.version,
                "expected_source_sha256": a.expected_source, **fields}
        temp = a.state.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2)); temp.replace(a.state)
        print(json.dumps(data), flush=True)
    api = connect()
    while True:
        try:
            status = api.kernels_status(a.kernel).to_dict()["status"]
        except Exception as exc:
            record(phase="status_retry", error_type=type(exc).__name__)
            time.sleep(30); continue
        record(phase="waiting", status=status)
        if status in {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
            break
        time.sleep(30)
    subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_api.py"), "output", a.kernel,
        "--out", str(a.output), "--pattern", a.pattern], check=True, cwd=ROOT)
    logs = "\n".join(path.read_text(errors="replace") for path in a.output.rglob("*.log"))
    if a.expected_source not in logs:
        record(phase="source_verification_failed", status=status)
        raise RuntimeError("Collected output does not identify the expected source snapshot")
    if status != "COMPLETE":
        record(phase="failed_run_evidence_collected", status=status); return
    if a.analysis_output:
        paths = list(a.output.rglob("session_status.json"))
        if len(paths) != 1:
            record(phase="missing_session_ledger", status=status)
            raise RuntimeError("Completed kernel lacks one unambiguous experiment ledger")
        process = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_segmentation.py"),
            str(paths[0].parent), "--output", str(a.analysis_output)], cwd=ROOT)
        if process.returncode:
            record(phase="analysis_failed", status=status, returncode=process.returncode); return
    record(phase="ready_for_review", status=status, output=str(a.output))


if __name__ == "__main__":
    main()
