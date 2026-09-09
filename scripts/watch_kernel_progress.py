#!/usr/bin/env python3
"""Read an existing Kaggle log stream; reconnect without submitting/cancelling jobs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from kaggle_api import connect


EPOCH = re.compile(r"\[(\w+)\]\s+epoch\s+(\d+)\s+loss\s+(\S+)\s+score\s+(\S+)\s+best\s+(\S+)\s+@\s+(-?\d+)")
OUTER = re.compile(r"\[(\w+) outer\] evaluated (\d+)/(\d+) volumes in ([\d.]+)s")
SUBSETS = re.compile(r"\[(\w+) robustness\] modality subsets (\d+)/(\d+) complete in ([\d.]+)s")
PROTECTED_START = re.compile(r"^START PROTECTED (\d+) (\w+) GPU (\d+)$")
PROTECTED_FULL = re.compile(r"\[protected (\w+) seed (\d+)\] evaluated (\d+)/(\d+) volumes in ([\d.]+)s")
PROTECTED_SUBSET = re.compile(r"\[protected (\w+) seed (\d+)\] subset (\d+)/15 saved; ([\d.]+)s this attempt")
PROTECTED_END = re.compile(r"^PROTECTED SESSION COMPLETE (\d+) /12$")
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}


def finite_or_none(value):
    number = float(value)
    return number if math.isfinite(number) else None


def observe_protected_line(state, line):
    """Return None for other logs, or whether protected progress advanced.

    Saved subsets and a session log are progress evidence. Completion still
    requires exported quantitative and qualitative receipts to be verified.
    """
    start = PROTECTED_START.search(line)
    full = PROTECTED_FULL.search(line)
    subset = PROTECTED_SUBSET.search(line)
    end = PROTECTED_END.search(line)
    if not any((start, full, subset, end)):
        return None
    if end:
        completed = int(end.group(1))
        if state.get("protected_session_logged_completed") == completed:
            return False
        state["protected_session_logged_completed"] = completed
        state["protected_planned_checkpoint_evaluations"] = 12
        return True
    runs = state.setdefault("protected_runs", {})
    if start:
        seed, name, gpu = start.groups()
        key = f"{name}_s{seed}"
        if key in runs:
            return False
        runs[key] = {"model": name, "seed": int(seed), "gpu": int(gpu), "phase": "started"}
        return True
    match = full or subset
    name, seed = match.group(1, 2)
    current = runs.setdefault(f"{name}_s{seed}", {"model": name, "seed": int(seed)})
    if full:
        done, total, seconds = match.group(3, 4, 5)
        if int(done) <= current.get("full_volumes_done", 0):
            return False
        current.update(full_volumes_done=int(done), full_volumes_total=int(total),
                       full_volumes_seconds=float(seconds))
        if not current.get("modality_subsets_done"):
            current["phase"] = "full_evaluation"
    else:
        done, seconds = match.group(3, 4)
        if int(done) <= current.get("modality_subsets_done", 0):
            return False
        current.update(modality_subsets_done=int(done), modality_subsets_total=15,
                       attempt_seconds=float(seconds),
                       phase="all_subsets_logged_saved" if int(done) == 15 else "subset_evaluation")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {"kernel": args.kernel, "expected_source_sha256": args.expected_source,
             "source_observed": False, "models": {}, "reconnections": 0,
             "interpretation": "Live progress only; final exported artifacts require independent verification."}
    if args.state.exists():
        previous = json.loads(args.state.read_text())
        if previous.get("kernel") != args.kernel or previous.get("expected_source_sha256") != args.expected_source:
            raise ValueError("Progress file belongs to a different kernel/source")
        state.update(previous)
    seen = set()
    api = connect()

    def save():
        state["observed_utc"] = datetime.now(timezone.utc).isoformat()
        temp = args.state.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2, allow_nan=False))
        temp.replace(args.state)

    while True:
        try:
            state["remote_status"] = api.kernels_status(args.kernel).to_dict()["status"]
            save()
            for event in api.kernels_logs_stream(args.kernel):
                data = str(event.get("data", ""))
                for line in data.splitlines():
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    if args.expected_source in line:
                        state["source_observed"] = True
                        print("EXPECTED SOURCE OBSERVED", args.kernel, flush=True)
                    protected_change = observe_protected_line(state, line)
                    if protected_change is not None:
                        if protected_change:
                            save()
                            print(line, flush=True)
                        continue
                    match = EPOCH.search(line)
                    if match:
                        name, epoch, loss, score, best, best_epoch = match.groups()
                        old = state["models"].get(name, {})
                        if old.get("phase") != "fit_and_evaluation_logged_complete" and int(epoch) > old.get("epoch_zero_based", -1):
                            state["models"][name] = {"phase": "training", "epoch_zero_based": int(epoch),
                                "training_loss": finite_or_none(loss), "inner_selection_score": finite_or_none(score),
                                "best_inner_score": finite_or_none(best),
                                "selected_epoch_zero_based": int(best_epoch) if int(best_epoch) >= 0 else None,
                                "log_time_seconds": event.get("time")}
                            save()
                            if score.lower() != "nan":
                                print(line, flush=True)
                    elif line.startswith("START ") or line.startswith("COMPLETED "):
                        name = line.split()[1]
                        phase = "started" if line.startswith("START ") else "fit_and_evaluation_logged_complete"
                        current = state["models"].setdefault(name, {})
                        if phase != "started" or not current:
                            current["phase"] = phase
                            save()
                            print(line, flush=True)
                    elif OUTER.search(line) or SUBSETS.search(line):
                        outer = OUTER.search(line)
                        inference = outer or SUBSETS.search(line)
                        name, done, total, seconds = inference.groups()
                        current = state["models"].setdefault(name, {})
                        if current.get("phase") == "fit_and_evaluation_logged_complete":
                            continue
                        prefix = "outer_volumes" if outer else "modality_subsets"
                        if int(done) <= current.get(prefix + "_done", 0):
                            continue
                        current.update({prefix + "_done": int(done), prefix + "_total": int(total),
                                        prefix + "_seconds": float(seconds)})
                        if not outer or current.get("phase") != "subset_evaluation":
                            current["phase"] = "outer_evaluation" if outer else "subset_evaluation"
                        save()
                        if outer or int(done) in {1, 5, 10, int(total)}:
                            print(line, flush=True)
                    elif any(marker in line for marker in ("Traceback", "RuntimeError", "CUDA out of memory")):
                        state["last_error_line"] = line
                        save()
                        print(line, flush=True)
            state["remote_status"] = api.kernels_status(args.kernel).to_dict()["status"]
            save()
            if state["remote_status"] in TERMINAL:
                print("REMOTE TERMINAL STATUS", state["remote_status"], args.kernel, flush=True)
                return
        except Exception as exc:
            # Exception text may contain request details; record only its class.
            state["last_stream_error_type"] = type(exc).__name__
            save()
            print("LOG STREAM RETRY", type(exc).__name__, args.kernel, flush=True)
        state["reconnections"] += 1
        save()
        time.sleep(20)


if __name__ == "__main__":
    main()
