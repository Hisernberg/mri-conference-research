"""QMMF-Net: compact arbitrary-modality 2.5D segmentation network.

Shared pseudo-3D modality stems -> multi-scale per-modality features ->
QMMF at every scale -> light large-kernel context mixer -> light U-Net decoder
with deep supervision -> three overlapping sigmoid outputs (WT, TC, ET).

Design targets from plan 7.7: < 8M parameters, < 14.5 GB peak VRAM on one T4.
Both are *gates to measure*, not claims; `count_parameters` and the VRAM
profiler in utils.py report the measured values into the experiment ledger.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MODALITIES, ModelConfig
from .blocks import ContextMixer, ConvBlock, PseudoConv3DStem, UpBlock, norm_layer
from .qmmf import build_fusion


class QMMFNet(nn.Module):
    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        n_modalities: int = 4,
        context_slices: int = 5,
        n_outputs: int = 3,
        fusion_kind: str = "qmmf",
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.n_modalities = n_modalities
        self.n_outputs = n_outputs
        self.widths = tuple(cfg.widths)
        self.deep_supervision = cfg.deep_supervision

        # One learned identity embedding per sequence; the stem itself is shared.
        self.modality_embedding = nn.Parameter(
            torch.randn(n_modalities, cfg.modality_embed_dim) * 0.02
        )

        self.stem = PseudoConv3DStem(
            context_slices, self.widths[0],
            cond_dim=cfg.modality_embed_dim, use_film=cfg.use_film,
        )

        # Per-modality encoder stages, weights shared across modalities.
        self.encoders = nn.ModuleList([
            ConvBlock(self.widths[i - 1], self.widths[i], depth=2, dropout=cfg.dropout)
            for i in range(1, len(self.widths))
        ])
        self.pool = nn.AvgPool2d(2)

        fusion_kwargs = dict(
            quality_dim=cfg.quality_dim,
            embed_dim=cfg.modality_embed_dim,
            hidden=cfg.gate_hidden,
            moments=cfg.moment_fusion,
            use_quality=cfg.use_quality,
            learned_gate=cfg.use_availability_gate,
        ) if fusion_kind == "qmmf" else dict(n_modalities=n_modalities)

        self.fusions = nn.ModuleList([
            build_fusion(fusion_kind, w, **fusion_kwargs) for w in self.widths
        ])

        self.context = ContextMixer(
            self.widths[-1], depth=cfg.context_depth,
            kernel=cfg.context_kernel, mode=cfg.context_block,
        )

        rev = list(reversed(self.widths))
        self.decoders = nn.ModuleList([
            UpBlock(rev[i], rev[i + 1], rev[i + 1], dropout=cfg.dropout)
            for i in range(len(rev) - 1)
        ])
        self.head = nn.Conv2d(self.widths[0], n_outputs, 1)
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(rev[i + 1], n_outputs, 1) for i in range(len(rev) - 2)]
        ) if cfg.deep_supervision else nn.ModuleList()

    # ------------------------------------------------------------------ #
    def encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Per-modality multi-scale features.

        image: [B, M, D, H, W] -> list of [B, M, C_s, H_s, W_s], one per scale.
        """
        b, m, d, h, w = image.shape
        flat = image.reshape(b * m, d, h, w)
        cond = self.modality_embedding.unsqueeze(0).expand(b, -1, -1).reshape(b * m, -1)

        feats: List[torch.Tensor] = []
        x = self.stem(flat, cond)
        feats.append(x)
        for enc in self.encoders:
            x = enc(self.pool(x))
            feats.append(x)
        return [f.reshape(b, m, *f.shape[1:]) for f in feats]

    def forward(
        self,
        image: torch.Tensor,             # [B, M, D, H, W]
        availability: torch.Tensor,      # [B, M]
        quality: torch.Tensor,           # [B, M, Q]
        return_aux: bool = False,
    ):
        if image.dim() != 5:
            raise ValueError(f"Expected [B, M, D, H, W]; got {tuple(image.shape)}")
        if (availability.sum(dim=1) < 1).any():
            raise ValueError("At least one modality must be available per sample.")

        per_modality = self.encode(image)

        fused: List[torch.Tensor] = []
        aux_all: List[Dict[str, torch.Tensor]] = []
        for feats, fusion in zip(per_modality, self.fusions):
            z, aux = fusion(feats, availability, quality, self.modality_embedding)
            fused.append(z)
            aux_all.append(aux)

        x = self.context(fused[-1])

        # Deep supervision heads exist only to shape the training gradient, so
        # they are evaluated in training mode only - at inference they would be
        # pure wasted compute on a T4.
        want_deep = self.deep_supervision and self.training
        deep_logits: List[torch.Tensor] = []
        skips = list(reversed(fused[:-1]))
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[i])
            if want_deep and i < len(self.aux_heads):
                deep_logits.append(self.aux_heads[i](x))
        logits = self.head(x)

        if not return_aux:
            return logits
        return logits, {
            "deep_logits": deep_logits,
            "gate_alpha": torch.stack([a["alpha"] for a in aux_all], dim=1),   # [B,S,M]
            "gate_entropy": torch.stack([a["gate_entropy"] for a in aux_all], dim=1),
            "fused": fused,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def gate_summary(
        self, image: torch.Tensor, availability: torch.Tensor, quality: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Per-modality gate weights averaged over scales, for the gate-Shapley
        alignment analysis (plan 10.4)."""
        _, aux = self.forward(image, availability, quality, return_aux=True)
        return {
            "alpha_per_scale": aux["gate_alpha"],
            "alpha_mean": aux["gate_alpha"].mean(dim=1),
            "entropy_mean": aux["gate_entropy"].mean(dim=1),
        }


# --------------------------------------------------------------------------- #
# Nested post-processing
# --------------------------------------------------------------------------- #
def project_nested(probs: torch.Tensor) -> torch.Tensor:
    """Enforce ET subset TC subset WT on probabilities.

    p_TC <- min(p_TC, p_WT); p_ET <- min(p_ET, p_TC). Applied at inference, and
    ablation A12 removes the matching training penalty to test its contribution.
    """
    wt = probs[:, 0:1]
    tc = torch.minimum(probs[:, 1:2], wt)
    et = torch.minimum(probs[:, 2:3], tc)
    return torch.cat([wt, tc, et], dim=1)


def project_nested_numpy(probs):
    import numpy as np
    wt = probs[0]
    tc = np.minimum(probs[1], wt)
    et = np.minimum(probs[2], tc)
    return np.stack([wt, tc, et], axis=0)
