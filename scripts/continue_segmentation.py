#!/usr/bin/env python3
"""Submit the reviewed pilot only after its private CPU preparation succeeds."""
import json
from pathlib import Path
import subprocess
import sys
import time
from kaggle_api import connect

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "dasshovon/mri-segmentation-preparation-v2"
TARGET = "dasshovon/mri-segmentation-qmmf-v2"
STATE = ROOT / "runtime/segmentation_orchestration.json"


def record(**fields):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps({"time_unix": time.time(), **fields}, indent=2))
    temp.replace(STATE)
    print(json.dumps(fields), flush=True)


def main():
    api = connect()
    while True:
        status = api.kernels_status(PREPARATION).to_dict()["status"]
        record(phase="waiting_for_cpu_preparation", preparation_status=status)
        if status == "COMPLETE":
            break
        if status in {"ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}:
            raise RuntimeError(f"Preparation did not complete: {status}")
        time.sleep(30)
    # Build from the tested source and submit only this task's private notebook.
    subprocess.run([sys.executable, str(ROOT / "scripts/build_notebooks.py"),
                    "segmentation", "--scope", "pilot"], cwd=ROOT, check=True)
    result = api.kernels_push(str(ROOT / "kaggle/segmentation"),
                              acc="NvidiaTeslaT4", timeout="6600")
    record(phase="pilot_submitted", kernel=TARGET, response=result.to_dict())


if __name__ == "__main__":
    main()
