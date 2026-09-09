"""Model registry and the A0-A17 ablation table (plan 9.3).

Each ablation is expressed as a set of dotted config overrides plus, where
needed, a different fusion module. Nothing is hard-coded inside the network:
an ablation that cannot be expressed as a config change is a design smell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import ExperimentConfig, ModelConfig
from .baselines import (
    HeMIS25D, SegResNet3DWrapper, SwinUNETRWrapper, UNet2D, UNet25D,
)
from .qmmf_net import QMMFNet, project_nested, project_nested_numpy

MODEL_REGISTRY: Dict[str, Callable] = {
    "qmmf_net": QMMFNet,
    "unet2d": UNet2D,          # B1
    "unet25d": UNet25D,        # B2
    "hemis25d": HeMIS25D,      # B3
    "segresnet3d": SegResNet3DWrapper,   # B4
    "swinunetr": SwinUNETRWrapper,       # B6, optional
}

BASELINE_IDS = {
    "B1": "unet2d", "B2": "unet25d", "B3": "hemis25d",
    "B4": "segresnet3d", "B5": "qmmf_net", "B6": "swinunetr",
}


@dataclass(frozen=True)
class Ablation:
    ident: str
    description: str
    question: str
    overrides: Dict[str, Any]
    fusion_kind: str = "qmmf"
    data_overrides: Dict[str, Any] = None            # type: ignore[assignment]
    loss_overrides: Dict[str, Any] = None            # type: ignore[assignment]
    train_overrides: Dict[str, Any] = None           # type: ignore[assignment]

    def apply(self, cfg: ExperimentConfig) -> ExperimentConfig:
        merged: Dict[str, Any] = {}
        for prefix, table in (
            ("net", self.overrides), ("data", self.data_overrides),
            ("loss", self.loss_overrides), ("train", self.train_overrides),
        ):
            for key, value in (table or {}).items():
                merged[f"{prefix}.{key}"] = value
        merged["ablation"] = self.ident
        return cfg.merged(merged)


ABLATIONS: Dict[str, Ablation] = {a.ident: a for a in [
    Ablation("A0", "Full QMMF-Net", "Reference", {}),
    Ablation("A1", "Early channel concatenation",
             "Does the entire modality-aware design matter?",
             {}, fusion_kind="concat"),
    Ablation("A2", "Equal masked mean only", "Do the weighted moments matter?",
             {}, fusion_kind="equal_mean"),
    Ablation("A3", "HeMIS equal masked mean + variance", "Foundational baseline",
             {}, fusion_kind="equal_mean_var"),
    Ablation("A4", "Availability-only gate",
             "Separates quality conditioning from dynamic availability",
             {"use_quality": False, "use_availability_gate": False}),
    Ablation("A5", "Remove quality vector", "Direct quality contribution",
             {"use_quality": False}),
    Ablation("A6", "Remove weighted variance",
             "Disagreement / complementarity contribution",
             {"moment_fusion": ["mean", "max"]}),
    Ablation("A7", "Remove max-available branch",
             "Localised strong-evidence contribution",
             {"moment_fusion": ["mean", "variance"]}),
    Ablation("A8", "Remove axial/large-kernel context", "Context mixer contribution",
             {"context_block": "none"}),
    Ablation("A9", "Kernel 3 vs kernel 7", "Receptive-field trade-off",
             {"context_block": "kernel3"}),
    Ablation("A10", "Remove subset consistency",
             "Robustness and label-efficiency contribution",
             {}, loss_overrides={"consistency": 0.0}),
    Ablation("A11", "Uniform modality sampling from epoch 1",
             "Curriculum contribution",
             {}, train_overrides={"modality_curriculum": "uniform"}),
    Ablation("A12", "Remove nesting loss", "Anatomical consistency contribution",
             {}, loss_overrides={"nested": 0.0}),
    Ablation("A13", "Remove boundary loss", "Surface-metric contribution",
             {}, loss_overrides={"boundary": 0.0}),
    Ablation("A14", "D = 1 / 3 / 5 / 7 slices", "2D-to-2.5D context trade-off",
             {}, data_overrides={"context_slices": 3}),
    Ablation("A15", "Remove modality embedding / FiLM",
             "Identity-conditioning contribution", {"use_film": False}),
    # Widths chosen by parameter_matched_widths() so that A16 sits within ~1%
    # of QMMF-Net's parameter count at the default widths. Re-derive and update
    # this list whenever the default widths change; the matching is asserted in
    # tests/test_model.py rather than assumed.
    Ablation("A16", "Parameter-matched widened HeMIS / simple fusion",
             "Controls for capacity",
             {"widths": [24, 48, 98, 164]}, fusion_kind="equal_mean_var"),
    Ablation("A17", "Shuffle quality vectors across patients",
             "Negative control for spurious quality conditioning", {}),
]}

# A17 is a data-side control: the flag is read by SliceDataset, not the network.
DATASET_FLAG_ABLATIONS = {"A17": {"shuffle_quality": True},
                          "A5": {"zero_quality": True}}

# A14 sweeps several depths; the config generator expands them.
A14_VARIANTS = (1, 3, 5, 7)


def build_model(cfg: ExperimentConfig):
    """Instantiate the model named by cfg.model with the ablation applied."""
    name = cfg.model
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Have {sorted(MODEL_REGISTRY)}.")
    ablation = ABLATIONS.get(cfg.ablation)
    fusion_kind = ablation.fusion_kind if ablation else "qmmf"
    kwargs = dict(
        cfg=cfg.net,
        n_modalities=len(cfg.data.modalities),
        context_slices=cfg.data.context_slices,
        n_outputs=len(cfg.data.outputs),
    )
    if name == "qmmf_net":
        kwargs["fusion_kind"] = fusion_kind
    return MODEL_REGISTRY[name](**kwargs)


def parameter_matched_widths(
    cfg: ExperimentConfig,
    fusion_kind: str,
    base_widths=(24, 48, 96, 160),
    tolerance: float = 0.05,
    max_scale: float = 1.6,
    channel_multiple: int = 2,
) -> Dict[str, Any]:
    """Find widths that give `fusion_kind` the same parameter count as QMMF-Net.

    Plan 9.2 requires a parameter-matched control: a simpler fusion must not be
    allowed to look worse merely because it is smaller. This searches a scale
    factor over the base widths and returns the closest match, so the A16 entry
    above is derived rather than guessed.
    """
    from ..utils import count_parameters
    from .qmmf_net import QMMFNet

    reference = count_parameters(build_model(cfg))["total"]
    if channel_multiple < 1:
        raise ValueError("channel_multiple must be positive")
    best: Optional[Dict[str, Any]] = None
    scale = 1.0
    seen = set()
    while scale <= max_scale:
        widths = [max(channel_multiple, int(round(w * scale / channel_multiple)) * channel_multiple)
                  for w in base_widths]
        scale += 0.005
        if tuple(widths) in seen:
            continue
        seen.add(tuple(widths))
        model = QMMFNet(
            cfg.merged({"net.widths": widths}).net,
            n_modalities=len(cfg.data.modalities),
            context_slices=cfg.data.context_slices,
            n_outputs=len(cfg.data.outputs),
            fusion_kind=fusion_kind,
        )
        total = count_parameters(model)["total"]
        ratio = total / reference
        if best is None or abs(ratio - 1) < abs(best["ratio"] - 1):
            best = {"widths": widths, "parameters": total, "ratio": ratio,
                    "reference_parameters": reference}
    assert best is not None
    best["within_tolerance"] = abs(best["ratio"] - 1) <= tolerance
    best["channel_multiple"] = channel_multiple
    best["tolerance"] = tolerance
    return best


__all__ = [
    "MODEL_REGISTRY", "BASELINE_IDS", "ABLATIONS", "Ablation", "build_model",
    "parameter_matched_widths",
    "QMMFNet", "UNet2D", "UNet25D", "HeMIS25D", "SegResNet3DWrapper",
    "SwinUNETRWrapper", "project_nested", "project_nested_numpy",
    "DATASET_FLAG_ABLATIONS", "A14_VARIANTS",
]
