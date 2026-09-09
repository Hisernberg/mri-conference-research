"""Deduplicated, grouped, repeated-seed classification benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.nn import functional as F

from classification.models import MSCANet2D, Plain2DCNN, ResNet2D18

VARIANTS = ["full", "no_attention", "no_multiscale", "no_attention_no_multiscale",
            "no_augmentation", "plain_cnn", "narrow_resnet_gn"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def discover_root(base):
    base = Path(base)
    if any(p.parent.name.lower() in {"yes", "no"} for p in base.rglob("*.jpg")):
        return base
    raise FileNotFoundError(f"No yes/no image dataset under {base}")


def audit_images(root, out):
    """Identify bytes/pixels exactly; conservatively group approximate copies."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, invalid = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"} or p.parent.name.lower() not in {"yes", "no"}:
            continue
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("L")
                a = np.asarray(im)
                tiny = np.asarray(im.resize((32, 32), Image.Resampling.BILINEAR), dtype=float)
                d = dctn(tiny, norm="ortho")[:8, :8].ravel()[1:]
                ph = sum(int(v) << i for i, v in enumerate(d > np.median(d)))
                rows.append({"relative_path": p.relative_to(root).as_posix(),
                    "label": p.parent.name.lower(), "y": int(p.parent.name.lower() == "yes"),
                    "byte_hash": sha(p.read_bytes()),
                    "image_id": sha(str(a.shape).encode() + a.tobytes()),
                    "width": im.width, "height": im.height, "phash": ph, "tiny": tiny.ravel()})
        except (OSError, ValueError) as exc:
            invalid.append({"relative_path": p.relative_to(root).as_posix(), "error": str(exc)})
    if not rows:
        raise ValueError("No readable labeled images")
    raw = pd.DataFrame(rows)
    conflicts = raw.groupby("image_id").y.nunique()
    bad = set(conflicts[conflicts > 1].index)
    clean = raw[~raw.image_id.isin(bad)].drop_duplicates("image_id").reset_index(drop=True)
    parent = list(range(len(clean)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    candidates = []
    for i in range(len(clean)):
        for j in range(i):
            distance = (int(clean.iloc[i].phash) ^ int(clean.iloc[j].phash)).bit_count()
            if distance > 8:
                continue
            a, b = clean.iloc[i].tiny, clean.iloc[j].tiny
            corr = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 0 else 0.0
            grouped = distance <= 4 and corr >= 0.98
            candidates.append({"image_a": clean.iloc[j].image_id, "image_b": clean.iloc[i].image_id,
                               "phash_distance": distance, "correlation": corr, "grouped": grouped})
            if grouped:
                parent[find(i)] = find(j)
    components = {}
    for i in range(len(clean)):
        components.setdefault(find(i), []).append(clean.iloc[i].image_id)
    clean["group_id"] = [min(components[find(i)]) for i in range(len(clean))]
    raw.drop(columns=["tiny"]).to_csv(out / "all_files.csv", index=False)
    clean = clean.drop(columns=["tiny"])
    clean.to_csv(out / "image_manifest.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "similarity_candidates.csv", index=False)
    report = {"raw_files": len(raw), "byte_unique": int(raw.byte_hash.nunique()),
              "decoded_unique_before_quarantine": int(raw.image_id.nunique()),
              "conflicting_exact_images_quarantined": len(bad), "images": len(clean),
              "similarity_groups": int(clean.group_id.nunique()),
              "class_counts": {str(k): int(v) for k, v in clean.label.value_counts().items()},
              "invalid_files": invalid, "grouping_rule": "pHash distance <= 4 and Pearson r >= 0.98 at 32x32",
              "group_unit": "image similarity; patient identity unknown",
              "manifest_sha256": sha(clean.to_csv(index=False).encode())}
    write_json(out / "dataset_audit.json", report)
    print("DATA AUDIT", json.dumps(report), flush=True)
    return clean, report


def locked_splits(df, n_folds=5, seed=42):
    splits, records = [], []
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (dev, test) in enumerate(outer.split(df, df.y, df.group_id)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1000 + fold)
        tr, va = next(inner.split(df.iloc[dev], df.iloc[dev].y, df.iloc[dev].group_id))
        parts = {"train": dev[tr], "validation": dev[va], "test": test}
        sets = [set(df.iloc[v].group_id) for v in parts.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert all(df.iloc[ix].y.nunique() == 2 for ix in parts.values()), "Class missing from partition"
        splits.append(parts)
        for role, indices in parts.items():
            for idx in indices:
                records.append({"fold": fold, "role": role, "image_id": df.iloc[idx].image_id,
                                "group_id": df.iloc[idx].group_id, "y": int(df.iloc[idx].y)})
    return splits, pd.DataFrame(records)


def load_tensor(root, df, size):
    images = []
    for rel in df.relative_path:
        with Image.open(Path(root) / rel) as im:
            im = ImageOps.exif_transpose(im).convert("L")
            im = ImageOps.pad(im, (size, size), method=Image.Resampling.BILINEAR, color=0)
            a = np.asarray(im, dtype=np.float32)
        mask = a > np.percentile(a, 10)
        if mask.sum() < 100:
            mask = np.ones_like(a, dtype=bool)
        a = np.clip((a - a[mask].mean()) / max(float(a[mask].std()), 1e-6), -5, 5)
        images.append(a)
    return torch.from_numpy(np.stack(images)[:, None])


def augment(x, generator):
    n = len(x)
    rand = lambda *s: torch.rand(*s, generator=generator, device=x.device)
    angle = (rand(n) * 2 - 1) * math.pi / 18
    flip = torch.where(rand(n) < 0.5, -1., 1.)
    t = torch.zeros(n, 2, 3, device=x.device)
    t[:, 0, 0], t[:, 0, 1] = angle.cos() * flip, -angle.sin()
    t[:, 1, 0], t[:, 1, 1] = angle.sin() * flip, angle.cos()
    grid = F.affine_grid(t, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return x * (0.9 + rand(n, 1, 1, 1) * 0.2) + (rand(n, 1, 1, 1) - 0.5) * 0.2


def model_for(name):
    if name == "plain_cnn":
        return Plain2DCNN(2)
    if name == "narrow_resnet_gn":
        return ResNet2D18(2)
    return MSCANet2D(2, use_attention="no_attention" not in name,
                    multiscale="no_multiscale" not in name)


def metrics(y, p):
    pred = np.asarray(p) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if np.sum(y) else None,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]))}


@torch.no_grad()
def predict(model, x, indices, batch=64):
    model.eval()
    return np.concatenate([model(x[ix]).float().softmax(1)[:, 1].cpu().numpy()
                           for ix in np.array_split(indices, max(1, math.ceil(len(indices) / batch)))])


def train_one(name, seed, fold, parts, x, y, df, args, split_hash):
    run_dir = args.out / "runs" / f"{name}_s{seed}_f{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "metrics.json"
    run_config = {"model": name, "seed": seed, "fold": fold, "epochs": args.epochs,
                  "patience": args.patience, "size": args.size, "batch": args.batch,
                  "split_hash": split_hash, "protocol": "classification-v2"}
    config_hash = sha(json.dumps(run_config, sort_keys=True).encode())
    if done.exists():
        prior = json.loads(done.read_text())
        if prior.get("config_hash") != config_hash:
            raise RuntimeError("Resume configuration differs from completed run")
        return prior, pd.read_csv(run_dir / "predictions.csv")
    random.seed(seed + fold * 1000); np.random.seed(seed + fold * 1000)
    torch.manual_seed(seed + fold * 1000); torch.cuda.manual_seed_all(seed + fold * 1000)
    model = model_for(name).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    counts = np.bincount(y[parts["train"]].cpu().numpy(), minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=x.device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    amp = x.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, best_loss, stale, history = -1., float("inf"), 0, []
    best_state, best_epoch = None, None
    start = time.monotonic()
    if amp:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng([seed, fold, epoch])
        indices = rng.permutation(parts["train"])
        aug_rng = torch.Generator(device=x.device).manual_seed(seed * 100000 + fold * 1000 + epoch)
        losses = []
        for begin in range(0, len(indices), args.batch):
            ix = indices[begin:begin + args.batch]
            bx = x[ix]
            if name != "no_augmentation":
                bx = augment(bx, aug_rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=x.device.type, enabled=amp):
                loss = criterion(model(bx), y[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classification loss")
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        p = predict(model, x, parts["validation"])
        m = metrics(y[parts["validation"]].cpu().numpy(), p)
        score, val_loss = m["macro_f1"], m["log_loss"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": score, "validation_log_loss": val_loss})
        if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and val_loss < best_loss - 1e-6):
            best_score, best_loss, stale, best_epoch = score, val_loss, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    pred = predict(model, x, parts["test"])
    frame = df.iloc[parts["test"]][["image_id", "group_id", "y"]].copy()
    frame["model"], frame["seed"], frame["fold"], frame["probability_yes"] = name, seed, fold, pred
    frame.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    ckpts = args.out / "checkpoints"; ckpts.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "config": run_config}, ckpts / f"{name}_s{seed}_f{fold}.pt")
    result = {**run_config, **metrics(frame.y.values, pred), "config_hash": config_hash,
              "best_epoch": best_epoch, "epochs_run": len(history),
              "inner_validation_macro_f1": best_score,
              "parameters": sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.monotonic() - start,
              "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if amp else 0.}
    write_json(done, result)
    print(json.dumps(result), flush=True)
    del model
    if amp:
        torch.cuda.empty_cache()
    return result, frame


def summarize(predictions, out, bootstrap=2000):
    """Descriptive group bootstrap of mean seed-wise OOF F1, never seed pseudoreplication."""
    rows = []
    for (name, seed), f in predictions.groupby(["model", "seed"]):
        rows.append({"model": name, "seed": int(seed), **metrics(f.y.values, f.probability_yes.values)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed_oof_metrics.csv", index=False)
    summary = per_seed.groupby("model").agg({k: ["mean", "std"] for k in metrics(np.array([0, 1]), np.array([0.2, 0.8]))})
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "model_summary.csv")
    # Identical image ordering makes paired resampling explicit and checkable.
    ids = sorted(predictions.image_id.unique())
    meta = predictions.drop_duplicates("image_id").set_index("image_id").loc[ids]
    groups = sorted(meta.group_id.unique())
    members = [np.where(meta.group_id.values == g)[0] for g in groups]
    model_names = sorted(predictions.model.unique())
    arrays = {}
    for name in model_names:
        matrix = predictions[predictions.model == name].pivot(index="image_id", columns="seed", values="probability_yes").reindex(ids)
        if matrix.isna().any().any():
            raise ValueError("Incomplete OOF matrix")
        arrays[name] = matrix.values >= 0.5
    y = meta.y.values

    def fast_f1(ix, name):
        pr, truth = arrays[name][ix], y[ix, None].astype(bool)
        tp = (pr & truth).sum(0); tn = (~pr & ~truth).sum(0)
        fp = (pr & ~truth).sum(0); fn = (~pr & truth).sum(0)
        return float(np.mean((2*tp / np.maximum(2*tp+fp+fn, 1) + 2*tn / np.maximum(2*tn+fp+fn, 1)) / 2))
    rng = np.random.default_rng(20260909)
    draws = {n: [] for n in model_names}
    for _ in range(bootstrap):
        ix = np.concatenate([members[g] for g in rng.integers(len(groups), size=len(groups))])
        for name in model_names:
            draws[name].append(fast_f1(ix, name))
    cis, contrasts = [], []
    for name in model_names:
        lo, hi = np.quantile(draws[name], [0.025, 0.975])
        cis.append({"model": name, "mean_seed_oof_macro_f1": fast_f1(np.arange(len(y)), name),
                    "ci_low": float(lo), "ci_high": float(hi), "bootstrap_groups": len(groups)})
        if name != "full":
            d = np.asarray(draws["full"]) - np.asarray(draws[name])
            lo, hi = np.quantile(d, [0.025, 0.975])
            contrasts.append({"comparison": f"full - {name}",
                              "macro_f1_delta": fast_f1(np.arange(len(y)), "full") - fast_f1(np.arange(len(y)), name),
                              "ci_low": float(lo), "ci_high": float(hi)})
    pd.DataFrame(cis).to_csv(out / "descriptive_group_bootstrap.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_ablation_deltas.csv", index=False)
    write_json(out / "inference_limitations.json", {"scope": "image-level exploratory benchmark",
        "bootstrap": "paired similarity groups; seeds averaged within each draw",
        "limitation": "unknown patients and dependent cross-validation fits; intervals are descriptive, not clinical or confirmatory significance"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.DataFrame(cis).sort_values("mean_seed_oof_macro_f1")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(c.mean_seed_oof_macro_f1, c.model,
                xerr=[c.mean_seed_oof_macro_f1 - c.ci_low, c.ci_high - c.mean_seed_oof_macro_f1], fmt="o", capsize=3)
    ax.set_xlabel("Mean seed-wise out-of-fold macro F1 (descriptive 95% group bootstrap)")
    ax.set_xlim(0, 1); fig.tight_layout()
    fig.savefig(out / "classification_comparison.png", dpi=250)
    fig.savefig(out / "classification_comparison.pdf")
    plt.close(fig)
    print(summary.round(4).to_string(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("classification_results"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    df, report = audit_images(args.root, args.out / "audit")
    parts, split_df = locked_splits(df)
    split_df.to_csv(args.out / "split_manifest.csv", index=False)
    split_hash = sha(split_df.to_csv(index=False).encode())
    write_json(args.out / "protocol_lock.json", {"split_sha256": split_hash,
        "dataset_sha256": report["manifest_sha256"], "models": args.models, "seeds": args.seeds,
        "epochs": args.epochs, "patience": args.patience, "size": args.size,
        "primary_metric": "macro_f1", "unit": "image similarity group; not patient",
        "preprocessing": "aspect-preserving letterbox, per-image foreground z-score"})
    if args.audit_only:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(args.out / "environment.json", {"python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None})
    x = load_tensor(args.root, df, args.size).to(device)
    y = torch.tensor(df.y.values, dtype=torch.long, device=device)
    results, frames = [], []
    for seed in args.seeds:
        for fold, partition in enumerate(parts):
            for name in args.models:
                r, f = train_one(name, seed, fold, partition, x, y, df, args, split_hash)
                results.append(r); frames.append(f)
                pd.DataFrame(results).to_csv(args.out / "per_run_metrics.csv", index=False)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.out / "oof_predictions.csv", index=False)
    summarize(predictions, args.out)
    write_json(args.out / "completion.json", {"completed": True, "runs": len(results),
        "expected_runs": len(args.seeds) * 5 * len(args.models),
        "scope": "repeated-seed image benchmark; conference/clinical validation not established"})


if __name__ == "__main__":
    main()
