#!/usr/bin/env python3
"""Audit effective auxiliary objectives without fitting or evaluating MRI cases."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation/src"))
sys.path.insert(0, str(ROOT / "segmentation/scripts"))
from conference_run import VARIANTS, save
from qmmf.config import ExperimentConfig
from qmmf.models import ABLATIONS, build_model


def audit():
    torch.set_num_threads(2); torch.manual_seed(20260909)
    snapshot = json.loads((ROOT / "audit/source_snapshots/segmentation_study_s42_v1.json").read_text())
    sources = snapshot["sources"]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if digest != snapshot["source_sha256"]:
        raise ValueError("Frozen main-source snapshot hash differs")
    for path, source in sources.items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"The effective-objective audit must use the submitted main implementation: {path}")
    image = torch.zeros(2,4,5,64,64)
    availability = torch.ones(2,4)
    quality = torch.zeros(2,4,7)
    rows = []
    for name, (model_name, ablation) in VARIANTS.items():
        cfg = ABLATIONS[ablation].apply(ExperimentConfig().merged({"model": model_name}))
        if name == "matched_moment_fusion":
            cfg = cfg.merged({"net.widths": [24,48,96,168]})
        model = build_model(cfg).train()
        with torch.no_grad():
            logits, aux = model(image, availability, quality, return_aux=True)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite architecture smoke output")
        heads = aux.get("deep_logits", [])
        rows.append({"variant": name, "model": model_name, "ablation": ablation,
            "parameters": sum(p.numel() for p in model.parameters()),
            "configured_deep_supervision_flag": cfg.net.deep_supervision,
            "effective_training_auxiliary_heads": len(heads),
            "auxiliary_head_shapes_on_synthetic_input": [list(h.shape) for h in heads],
            "effective_deep_supervision_weights": list(cfg.loss.deep_supervision_weights)[:len(heads)],
            "consistency_coefficient": cfg.loss.consistency,
            "interpretation": "reference model" if name == "qmmf" else "same-family component control" if model_name == "qmmf_net" else "cross-family system comparator"})
    dest = ROOT / "results/implementation_audit"; dest.mkdir(parents=True, exist_ok=True)
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "parent_source_sha256": digest, "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "procedure": "Untrained architectures in training mode; synthetic zero input; no MRI loaded, no optimization and no performance score computed.",
        "synthetic_input_shape": list(image.shape), "variants": rows,
        "finding": "Baseline classes return no auxiliary logits despite the shared configuration flag. QMMF-family variants expose two auxiliary heads.",
        "claim_limit": "HeMIS/U-Net contrasts compare architecture and effective auxiliary objective together; quality-specific attribution requires same-family controls.",
        "protected_test_opened": False}
    save(dest / "effective_objectives.json", record)
    pd.DataFrame(rows).to_csv(dest / "effective_objectives.csv", index=False)
    # Real pilot loss histories independently confirm which auxiliary term was logged.
    history = pd.read_csv(ROOT / "results/segmentation_grouped_pilot/training_history.csv")
    observed = []
    for name, frame in history.groupby("model"):
        present = frame.deep_supervision.notna()
        expected = next(row["effective_training_auxiliary_heads"] > 0 for row in rows if row["variant"] == name)
        if not (present.all() if expected else not present.any()):
            raise ValueError("Pilot loss history differs from the architecture audit")
        observed.append({"variant": name, "recorded_epochs": len(frame),
            "epochs_with_deep_supervision_term": int(present.sum()),
            "mean_deep_supervision_term": float(frame.deep_supervision.mean()) if expected else None})
    save(dest / "pilot_objective_confirmation.json", {"observations": observed,
        "history_sha256": hashlib.sha256((ROOT / "results/segmentation_grouped_pilot/training_history.csv").read_bytes()).hexdigest(),
        "not_a_new_training_experiment": True})
    print(pd.DataFrame(rows)[["variant","parameters","effective_training_auxiliary_heads","consistency_coefficient"]].to_string(index=False))


if __name__ == "__main__":
    audit()
