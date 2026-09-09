#!/usr/bin/env python3
"""Kaggle CPU staging job for an exactly equivalent, faster training read format."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from qmmf.slab_cache import build_training_slabs


def run(input_dir="/kaggle/input", output_dir="/kaggle/working/training_slabs"):
    start = time.monotonic()
    hits = list(Path(input_dir).rglob("preparation_complete.json"))
    if len(hits) != 1:
        raise ValueError("Attach exactly one verified original cache source")
    prepared = hits[0].parent
    report = json.loads((prepared / "dataset_fingerprint.json").read_text())
    manifest = pd.read_csv(prepared / "manifest.csv", dtype=str)
    ids = sorted(manifest.case_id.tolist())
    if len(ids) != 484 or len(set(ids)) != len(ids):
        raise ValueError("Unexpected input case universe")
    out = Path(output_dir); (out / "cache").mkdir(parents=True, exist_ok=True)
    def convert(cid):
        return build_training_slabs(prepared / "cache_v2" / f"{cid}.npz",
                                    out / "cache" / f"{cid}.npz", slab_depth=8)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(convert, ids), 1):
            rows.append(result)
            if i % 25 == 0:
                print(f"Verified lossless slabs {i}/{len(ids)}", flush=True)
    table = out / "case_inventory.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    complete = {"completed": True, "n_cases": len(rows), "manifest_hash": report["manifest_hash"],
        "source_cache_version": 2, "training_slab_format_version": 1, "slab_depth": 8,
        "voxel_equality_verified_for_all_cases": all(r["voxel_equality_verified"] for r in rows),
        "case_inventory_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic() - start, "slab_bytes": sum(r["slab_bytes"] for r in rows),
        "source_bytes": sum(r["source_bytes"] for r in rows),
        "purpose": "lossless selective decompression; no training, tuning or model evaluation"}
    (out / "training_slabs_complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    run()
