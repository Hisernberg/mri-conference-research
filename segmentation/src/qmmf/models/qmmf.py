"""Quality-conditioned masked moment fusion (QMMF).

Implements plan 7.4 / Appendix B exactly:

    h_m^s     = [GAP(F_m^s), q_m, e_m]
    l_m^s     = g_s(h_m^s)
    alpha_m^s = masked_softmax(l_m^s ; a_m)
    mu^s      = sum_m alpha_m^s * F_m^s
    v^s       = sum_m alpha_m^s * (F_m^s - mu^s)^2
    r^s       = max over available modalities of F_m^s
    Z^s       = Conv1x1([mu^s, sqrt(v^s + eps), r^s])

Two properties are guaranteed structurally, not by convention:
  1. an absent modality contributes nothing - it is excluded from the softmax
     denominator and from the max branch, so its features cannot leak in;
  2. the weights sum to one over the *available* modalities only, so the fused
     statistic keeps the same scale regardless of how many sequences are present.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e4      # finite so that fp16 autocast stays well-defined


def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides `channels`."""
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1
                   ) -> torch.Tensor:
    """Softmax over `dim`, restricted to entries where mask > 0.

    logits: [B, M, ...]; mask: [B, M] broadcast over the trailing dims.
    Rows with no available entry would be undefined; the caller guarantees
    sum(mask) >= 1 (see subsets.availability_mask), and this function still
    returns zeros rather than NaN if that guarantee is ever violated.
    """
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(logits.dtype)
    logits = logits.float()
    masked = logits.masked_fill(mask <= 0, NEG_INF)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * (mask > 0).to(weights.dtype)
    total = weights.sum(dim=dim, keepdim=True)
    return weights / total.clamp_min(1e-8)


def masked_feature_max(
    features: torch.Tensor, mask: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """Max over available modalities only. features: [B, M, C, H, W]."""
    while mask.dim() < features.dim():
        mask = mask.unsqueeze(-1)
    filled = features.masked_fill(mask <= 0, NEG_INF)
    out = filled.max(dim=dim).values
    # If (impossibly) nothing is available, fall back to zeros rather than -inf.
    any_avail = (mask > 0).any(dim=dim)
    return torch.where(any_avail.expand_as(out), out, torch.zeros_like(out))


class QMMFBlock(nn.Module):
    """One fusion block, applied at every encoder scale.

    Args:
        channels: feature channels C at this scale.
        quality_dim: length of the per-modality quality vector q_m.
        embed_dim: length of the learned modality identity embedding e_m.
        moments: which branches to concatenate before the 1x1 fusion.
        use_quality: A5 sets False (availability gating only).
        use_availability_gate: A4 keeps availability but drops learned gating.
    """

    def __init__(
        self,
        channels: int,
        quality_dim: int = 7,
        embed_dim: int = 16,
        hidden: int = 32,
        moments: Sequence[str] = ("mean", "variance", "max"),
        use_quality: bool = True,
        learned_gate: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.moments = tuple(moments)
        self.use_quality = use_quality
        self.learned_gate = learned_gate
        for m in self.moments:
            if m not in ("mean", "variance", "max"):
                raise KeyError(f"Unknown moment branch '{m}'")
        if "mean" not in self.moments:
            raise ValueError("The weighted mean branch is required.")

        descriptor_dim = channels + (quality_dim if use_quality else 0) + embed_dim
        if learned_gate:
            # No bias on the final layer: the gate logits enter a softmax over
            # modalities, so a shared additive constant is unidentifiable and
            # would sit in the model with a permanently zero gradient.
            self.gate = nn.Sequential(
                nn.Linear(descriptor_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1, bias=False),
            )
        else:
            self.gate = None

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.moments), channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )

    # ------------------------------------------------------------------ #
    def gate_logits(
        self,
        features: torch.Tensor,          # [B, M, C, H, W]
        quality: torch.Tensor,           # [B, M, Q]
        embedding: torch.Tensor,         # [M, E] or [B, M, E]
    ) -> torch.Tensor:
        """Per-modality scalar logit l_m from a pooled descriptor."""
        b, m = features.shape[0], features.shape[1]
        if self.gate is None:
            # A4: availability-only gate = equal weights over available inputs.
            return torch.zeros(b, m, device=features.device, dtype=features.dtype)
        gap = features.mean(dim=(-2, -1))                       # [B, M, C]
        parts = [gap]
        if self.use_quality:
            parts.append(quality.to(gap.dtype))
        emb = embedding if embedding.dim() == 3 else embedding.unsqueeze(0).expand(b, -1, -1)
        parts.append(emb.to(gap.dtype))
        descriptor = torch.cat(parts, dim=-1)                   # [B, M, D]
        return self.gate(descriptor).squeeze(-1)                # [B, M]

    def forward(
        self,
        features: torch.Tensor,          # [B, M, C, H, W]
        availability: torch.Tensor,      # [B, M]
        quality: torch.Tensor,           # [B, M, Q]
        embedding: torch.Tensor,         # [M, E]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.gate_logits(features, quality, embedding)
        alpha = masked_softmax(logits, availability, dim=1)      # [B, M]
        alpha_map = alpha[..., None, None, None]                 # [B, M, 1, 1, 1]

        features = features.float()  # stable moment reductions under autocast
        mu = (alpha_map * features).sum(dim=1)                   # [B, C, H, W]
        branches: List[torch.Tensor] = []
        for name in self.moments:
            if name == "mean":
                branches.append(mu)
            elif name == "variance":
                var = (alpha_map * (features - mu.unsqueeze(1)) ** 2).sum(dim=1)
                branches.append(torch.sqrt(var.clamp_min(0.0) + 1e-6))
            elif name == "max":
                branches.append(masked_feature_max(features, availability, dim=1))

        fused = self.fuse(torch.cat(branches, dim=1))
        aux = {
            "alpha": alpha,                                      # gate weights
            "gate_logits": logits,
            "gate_entropy": _entropy(alpha),
        }
        return fused, aux


def _entropy(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of the gate distribution, a collapse and uncertainty diagnostic."""
    p = alpha.float().clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Ablation variants A1 / A2 / A3 (plan 9.3)
# --------------------------------------------------------------------------- #
class ChannelConcatFusion(nn.Module):
    """A1: early channel concatenation with a zero-filled absent modality.

    The classic weak baseline: absent sequences are zeros and the network is
    never told which ones they are.
    """

    def __init__(self, channels: int, n_modalities: int = 4, **_: object):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * n_modalities, channels, 1, bias=False),
            _gn(channels), nn.GELU(),
        )

    def forward(self, features, availability, quality, embedding):
        b, m, c, h, w = features.shape
        masked = features * availability[:, :, None, None, None]
        fused = self.fuse(masked.reshape(b, m * c, h, w))
        return fused, {"alpha": availability / availability.sum(1, keepdim=True).clamp_min(1),
                       "gate_logits": torch.zeros(b, m, device=features.device),
                       "gate_entropy": torch.zeros(b, device=features.device)}


class EqualMaskedMean(nn.Module):
    """A2/A3: equal-weight masked mean (A2) or mean+variance (A3, HeMIS-style)."""

    def __init__(self, channels: int, with_variance: bool = False, **_: object):
        super().__init__()
        self.with_variance = with_variance
        n = 2 if with_variance else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * n, channels, 1, bias=False),
            _gn(channels), nn.GELU(),
        )

    def forward(self, features, availability, quality, embedding):
        a = availability
        w = a / a.sum(dim=1, keepdim=True).clamp_min(1e-8)
        wm = w[..., None, None, None]
        mu = (wm * features).sum(dim=1)
        parts = [mu]
        if self.with_variance:
            var = (wm * (features - mu.unsqueeze(1)) ** 2).sum(dim=1)
            parts.append(torch.sqrt(var.clamp_min(0) + 1e-6))
        fused = self.fuse(torch.cat(parts, dim=1))
        return fused, {"alpha": w, "gate_logits": torch.zeros_like(w),
                       "gate_entropy": _entropy(w)}


def build_fusion(kind: str, channels: int, **kwargs) -> nn.Module:
    """Fusion factory used by the ablation registry."""
    if kind == "qmmf":
        return QMMFBlock(channels, **kwargs)
    if kind == "concat":
        return ChannelConcatFusion(channels, **kwargs)
    if kind == "equal_mean":
        return EqualMaskedMean(channels, with_variance=False)
    if kind == "equal_mean_var":
        return EqualMaskedMean(channels, with_variance=True)
    raise KeyError(f"Unknown fusion kind '{kind}'")
