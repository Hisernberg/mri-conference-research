"""Reusable network blocks: pseudo-3D stem, FiLM, factorized large-kernel context.

No quadratic self-attention and no custom CUDA kernels anywhere on the
mandatory path (plan 7.6), so the model trains on a single 16 GB T4.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm: stable at the small per-GPU batches this study uses, and it
    needs no SyncBatchNorm across the two independent T4 jobs.

    The group count falls back to the largest divisor of `channels` that is at
    most `groups`, so an arbitrary width (e.g. a parameter-matched ablation)
    does not crash the model.
    """
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ConvBlock(nn.Module):
    """Conv -> Norm -> GELU, optionally twice."""

    def __init__(self, cin: int, cout: int, kernel: int = 3, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        c = cin
        for _ in range(depth):
            layers += [
                nn.Conv2d(c, cout, kernel, padding=kernel // 2, bias=False),
                norm_layer(cout),
                nn.GELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            c = cout
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector.

    Used to tell one shared stem which sequence it is looking at (plan 7.2),
    so the parameter count does not scale with the modality count.
    """

    def __init__(self, cond_dim: int, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels)
        )
        # Start as the identity so an untrained FiLM does not distort the stem.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return x * (1.0 + gamma) + beta


class PseudoConv3DStem(nn.Module):
    """Mix the D adjacent slices, then act in-plane.

    The depth mixing is a small factorized convolution over the slice axis
    (D -> 1), which is cheaper than a full 3D stem and keeps the model 2.5D.
    Weights are shared across modalities; identity comes from FiLM.
    """

    def __init__(self, context_slices: int, out_channels: int,
                 cond_dim: int = 16, use_film: bool = True):
        super().__init__()
        self.context_slices = context_slices
        self.depth_mix = nn.Conv3d(
            1, out_channels, kernel_size=(context_slices, 3, 3),
            padding=(0, 1, 1), bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.GELU()
        self.use_film = use_film
        self.film = FiLM(cond_dim, out_channels) if use_film else None
        self.refine = ConvBlock(out_channels, out_channels, depth=1)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """x: [B, D, H, W] for one modality -> [B, C, H, W]."""
        h = self.depth_mix(x.unsqueeze(1)).squeeze(2)      # [B, C, H, W]
        h = self.act(self.norm(h))
        if self.use_film and cond is not None:
            h = self.film(h, cond)
        return self.refine(h)


class FactorizedLargeKernelBlock(nn.Module):
    """Depthwise (1 x k) then (k x 1) large-kernel convolution with a pointwise MLP.

    Gives a wide receptive field with linear cost, which is the cheap stand-in
    for global attention at the bottleneck (plan 7.6). Ablation A8 removes the
    block; A9 replaces k=7 with k=3.
    """

    def __init__(self, channels: int, kernel: int = 7, expansion: int = 2):
        super().__init__()
        pad = kernel // 2
        self.dw_h = nn.Conv2d(channels, channels, (1, kernel), padding=(0, pad),
                              groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, (kernel, 1), padding=(pad, 0),
                              groups=channels, bias=False)
        self.norm = norm_layer(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1), nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )
        # Small non-zero LayerScale: zero-init would give the block's own
        # weights exactly zero gradient on the first step.
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_v(self.dw_h(x))
        h = self.mlp(self.norm(h))
        return x + self.gamma * h


class ContextMixer(nn.Module):
    """Two to four lightweight large-kernel blocks at the bottleneck."""

    def __init__(self, channels: int, depth: int = 2, kernel: int = 7,
                 mode: str = "factorized_large_kernel"):
        super().__init__()
        if mode == "none":
            self.body = nn.Identity()
        elif mode == "kernel3":
            self.body = nn.Sequential(
                *[FactorizedLargeKernelBlock(channels, 3) for _ in range(depth)]
            )
        elif mode == "factorized_large_kernel":
            self.body = nn.Sequential(
                *[FactorizedLargeKernelBlock(channels, kernel) for _ in range(depth)]
            )
        else:
            raise KeyError(f"Unknown context block '{mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpBlock(nn.Module):
    """Bilinear upsample + skip concatenation + conv (light U-Net decoder)."""

    def __init__(self, cin: int, cskip: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1, bias=False)
        self.block = ConvBlock(cout + cskip, cout, depth=2, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))
