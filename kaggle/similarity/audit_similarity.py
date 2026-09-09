#!/usr/bin/env python3
"""CPU audit of scale-invariant volume similarity; never trains or evaluates a model."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def pearson_matrix(matrix):
    # Long float32 sums biased the normalization enough to exceed correlation 1.
    # Use float64 for centering, norms and products on the sampled signatures.
    values = np.array(matrix, dtype=np.float64, copy=True)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("A modality has a degenerate spatial signature")
    values /= norms[:, None]
    result = values @ values.T
    if not np.allclose(np.diag(result), 1., rtol=0., atol=1e-10):
        raise FloatingPointError("Correlation normalization failed")
    return np.clip(result, -1., 1.)


def run(input_dir="/kaggle/input", output_dir="/kaggle/working/segmentation_similarity"):
    start = time.monotonic()
    hits = list(Path(input_dir).rglob("preparation_complete.json"))
    if len(hits) != 1:
        raise ValueError(f"Expected one prepared source, found {len(hits)}")
    prepared = hits[0].parent
    manifest = pd.read_csv(prepared / "manifest.csv", dtype=str).sort_values("case_id")
    report = json.loads((prepared / "dataset_fingerprint.json").read_text())
    ids = manifest.case_id.tolist()
    if len(ids) != 484 or len(set(ids)) != len(ids):
        raise ValueError("Unexpected case universe")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stride, correlation_cutoff, support_cutoff = 4, .995, .995
    shape = (240, 240, 155)
    axes = [np.arange(0, s, stride) for s in shape]
    sample_shape = tuple(len(a) for a in axes)
    size = int(np.prod(sample_shape))
    vectors = np.zeros((len(ids), 4, size), np.float32)
    support = np.zeros((len(ids), size), np.float32)

    def sample(item):
        index, cid = item
        with np.load(prepared / "cache_v2" / f"{cid}.npz", allow_pickle=False) as z:
            if int(z["cache_version"]) != 2 or tuple(z["original_shape"]) != shape:
                raise ValueError(f"Unexpected cache geometry/version: {cid}")
            if list(z["source_channel_order"]) != [1, 2, 3, 0]:
                raise ValueError(f"Unexpected channel order: {cid}")
            box = z["crop_box"].reshape(3, 2)
            destination = [np.flatnonzero((a >= lo) & (a < hi)) for a, (lo, hi) in zip(axes, box)]
            source = [a[valid] - lo for a, valid, (lo, _) in zip(axes, destination, box)]
            image, mask = z["image"], z["brain_mask"]
            v = vectors[index].reshape((4,) + sample_shape)
            v[np.ix_(np.arange(4), *destination)] = image[np.ix_(np.arange(4), *source)]
            support[index].reshape(sample_shape)[np.ix_(*destination)] = mask[np.ix_(*source)]
        if not np.isfinite(vectors[index]).all():
            raise ValueError(f"Non-finite cache values: {cid}")
        return index

    with ThreadPoolExecutor(max_workers=2) as pool:
        for n, _ in enumerate(pool.map(sample, enumerate(ids)), 1):
            if n % 25 == 0:
                print(f"Sampled {n}/{len(ids)} volumes", flush=True)
    # Pearson correlations on the same original-coordinate grid. This is
    # independent of the labels and insensitive to per-sequence intensity scale.
    with threadpool_limits(limits=2):
        corr = np.stack([pearson_matrix(vectors[:, m]) for m in range(4)])
        overlap = support @ support.T
    counts = support.sum(axis=1)
    dice = 2 * overlap / np.maximum(counts[:, None] + counts[None, :], 1)
    high = (corr.min(axis=0) >= correlation_cutoff) & (dice >= support_cutoff)
    lookup = {cid: i for i, cid in enumerate(ids)}
    candidates = {tuple(sorted((p["a"], p["b"]))): p["hamming"]
                  for p in report["duplicates"]["near_duplicate_pairs"]}
    for i, j in zip(*np.where(np.triu(high, k=1))):
        candidates.setdefault((ids[i], ids[j]), None)
    # Keep every coarse candidate together as a conservative anti-leakage rule;
    # high-correlation pairs can add edges missed by that initial screen.
    parent = list(range(len(ids)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    rows = []
    for (a, b), hamming in sorted(candidates.items()):
        i, j = lookup[a], lookup[b]
        parent[root(j)] = root(i)
        rows.append({"case_a": a, "case_b": b, "coarse_hamming": hamming,
            "correlation_t1": float(corr[0, i, j]), "correlation_t1ce": float(corr[1, i, j]),
            "correlation_t2": float(corr[2, i, j]), "correlation_flair": float(corr[3, i, j]),
            "support_dice": float(dice[i, j]), "high_similarity": bool(high[i, j]),
            "same_annotation_hash": manifest.iloc[i].label_array_sha256 == manifest.iloc[j].label_array_sha256})
    components = {}
    for i, cid in enumerate(ids):
        components.setdefault(root(i), []).append(cid)
    groups = {"sg_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]: members
              for members in sorted(components.values())}
    mapping = {cid: g for g, members in groups.items() for cid in members}
    table = pd.DataFrame(rows)
    table.to_csv(out / "similarity_pairs.csv", index=False)
    grouping = {"groups": groups, "case_to_group": mapping,
                "rule": "connected components of all coarse Hamming <= 4 candidates plus all four-modality Pearson >= .995 and support Dice >= .995 pairs",
                "interpretation": "conservative image-similarity groups, not verified patient identifiers"}
    (out / "case_groups.json").write_text(json.dumps(grouping, indent=2, sort_keys=True))
    summary = {"completed": True, "n_cases": len(ids), "n_groups": len(groups),
        "n_coarse_candidate_pairs": len(report["duplicates"]["near_duplicate_pairs"]),
        "n_high_similarity_pairs": int(table.high_similarity.sum()),
        "n_added_high_similarity_pairs": int(table.coarse_hamming.isna().sum()),
        "largest_group": max(map(len, groups.values())), "manifest_hash": report["manifest_hash"],
        "sampling_stride_voxels": stride, "correlation_cutoff_all_four": correlation_cutoff,
        "support_dice_cutoff": support_cutoff, "wall_seconds": time.monotonic() - start,
        "correlation_dtype": "float64", "model_outcomes_used": False}
    (out / "completion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
