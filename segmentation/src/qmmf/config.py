"""Typed experiment configuration for QMMF-Net.

Mirrors Appendix A of the master research plan. Every field that changes a
result must live here so that `config_hash` identifies a run uniquely.

Nothing in this module reads data or touches the GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_VERSION = "qmmf-v2.0-corrected"

# Canonical modality order. Never assume this from filenames: Notebook 01
# verifies it against dataset.json and intensity checks (plan 5.5).
MODALITIES: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
OUTPUTS: Tuple[str, ...] = ("wt", "tc", "et")


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    crop_size: Tuple[int, int] = (192, 192)
    context_slices: int = 5           # D in the plan; ablation A14 uses 1/3/5/7
    modalities: Tuple[str, ...] = MODALITIES
    outputs: Tuple[str, ...] = OUTPUTS
    normalization: str = "robust_zscore_nonzero_brain"
    clip_sigma: float = 5.0
    # Sampling policy, plan 6.3
    p_tumor_window: float = 0.50
    p_boundary_window: float = 0.25
    p_random_window: float = 0.25
    # Augmentation, plan 6.4
    aug_flip_lr: bool = True
    aug_rotation_deg: float = 10.0
    aug_scale: float = 0.1
    aug_elastic: bool = False         # only after a pilot confirms label fidelity
    aug_intensity: bool = True
    # Label efficiency, plan 6.5
    label_fraction: float = 1.0       # 0.25 / 0.50 / 1.0, nested subsets
    # Integrity, filled by Notebook 01
    manifest_hash: str = ""
    split_hash: str = ""
    root: str = ""


@dataclass
class ModelConfig:
    name: str = "qmmf_net"
    widths: Tuple[int, ...] = (24, 48, 96, 160)
    moment_fusion: Tuple[str, ...] = ("mean", "variance", "max")
    quality_dim: int = 7
    modality_embed_dim: int = 16
    context_block: str = "factorized_large_kernel"   # or "none" (A8), "kernel3" (A9)
    context_depth: int = 2
    context_kernel: int = 7
    deep_supervision: bool = True
    gate_hidden: int = 32
    use_quality: bool = True          # A5 sets False
    use_availability_gate: bool = True  # A4 keeps this, drops quality
    dropout: float = 0.0
    use_film: bool = True             # A15 sets False


@dataclass
class LossConfig:
    dice: float = 0.60
    bce: float = 0.40
    seg: float = 1.00
    boundary: float = 0.10
    nested: float = 0.05
    consistency: float = 0.20
    calibration: float = 0.00         # optional locked ablation, not assumed useful
    deep_supervision_weights: Tuple[float, ...] = (0.5, 0.25)
    consistency_warmup_frac: float = 0.15
    consistency_conf_threshold: float = 0.90


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4                  # pilot range 2e-4..5e-4
    weight_decay: float = 1e-3        # pilot range 1e-4..1e-2
    amp: bool = True
    batch_size: int = 4               # per-GPU; effective batch via accumulation
    accumulation_steps: int = 4       # effective ~16-32 central slices
    max_epochs: int = 120
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    ema_decay_start: float = 0.99
    ema_decay_end: float = 0.999
    modality_curriculum: str = "v1"   # A11 sets "uniform"
    num_workers: int = 2
    persistent_workers: bool = False
    seed: int = 42
    steps_per_epoch: int = 250
    val_every: int = 1
    grad_checkpointing: bool = False


@dataclass
class EvalConfig:
    nsd_tolerance_mm: float = 1.0
    empty_reference_rule: str = "exclude_from_dice_report_separately"
    bootstrap_replicates: int = 10_000
    alpha: float = 0.05
    non_inferiority_margin_pp: float = 1.5   # H1, locked before any test result
    calibration_bins: int = 20
    selective_coverages: Tuple[float, ...] = (1.0, 0.9, 0.8)
    mc_seeds: Tuple[int, ...] = (42, 43, 44)  # three-seed deep ensemble


@dataclass
class ExperimentConfig:
    protocol_version: str = PROTOCOL_VERSION
    seed: int = 42
    fold: int = 0
    label_fraction: float = 1.0
    model: str = "qmmf_net"
    ablation: str = "A0"
    stage: str = "development"        # development | locked_adult | ood_pediatric
    data: DataConfig = field(default_factory=DataConfig)
    net: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "/kaggle/working/qmmf_runs"
    notes: str = ""

    # ----------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return _tuples_to_lists(asdict(self))

    def config_hash(self) -> str:
        """Stable hash of the fully resolved configuration (Appendix C)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def run_id(self) -> str:
        return (
            f"{self.model}_{self.ablation}_f{self.fold}_s{self.seed}"
            f"_lf{int(self.label_fraction * 100)}_{self.config_hash()[:8]}"
        )

    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    # ----------------------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        sub = {
            "data": DataConfig,
            "net": ModelConfig,
            "loss": LossConfig,
            "train": TrainingConfig,
            "evaluation": EvalConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in sub.items():
            kwargs[key] = _build(klass, d.pop(key, {}) or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        kwargs.update(d)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml  # local import keeps the dependency optional
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        raw = raw.get("experiment", raw) if isinstance(raw, dict) else raw
        return cls.from_dict(raw)

    def merged(self, overrides: Dict[str, Any]) -> "ExperimentConfig":
        """Return a copy with dotted-key overrides applied, e.g. {'net.widths': [...]}"""
        base = self.to_dict()
        for dotted, value in overrides.items():
            node = base
            parts = dotted.split(".")
            for p in parts[:-1]:
                if p not in node:
                    raise KeyError(f"Unknown config section '{p}' in '{dotted}'")
                node = node[p]
            if parts[-1] not in node:
                raise KeyError(f"Unknown config key '{dotted}'")
            node[parts[-1]] = value
        return ExperimentConfig.from_dict(base)


# --------------------------------------------------------------------------- #
def _build(klass, d: Dict[str, Any]):
    fields = {f.name: f for f in dataclasses.fields(klass)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"Unknown keys for {klass.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in d.items():
        anno = str(fields[name].type)
        if "Tuple" in anno and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return klass(**kwargs)


def _tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(v) for v in obj]
    return obj


__all__ = [
    "ExperimentConfig", "DataConfig", "ModelConfig", "LossConfig",
    "TrainingConfig", "EvalConfig", "MODALITIES", "OUTPUTS", "PROTOCOL_VERSION",
]
