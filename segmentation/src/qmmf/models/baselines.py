"""Reimplemented baselines B1-B4 (+ optional B6/B7), plan 9.1.

Every baseline takes the *same* input signature as QMMF-Net
(image, availability, quality) so that the training loop, augmentation, splits,
metric code and modality-subset curriculum are byte-for-byte identical across
models. That is what makes the comparison fair (plan 9.2); differences in
tuning budget are disclosed rather than hidden.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .blocks import ConvBlock, PseudoConv3DStem, UpBlock, norm_layer
from .qmmf import EqualMaskedMean, masked_feature_max


class _UNet2DCore(nn.Module):
    """Shared encoder/decoder body so baselines differ only in their input stage."""

    def __init__(self, cin: int, widths: Sequence[int], n_outputs: int = 3):
        super().__init__()
        self.widths = tuple(widths)
        self.inc = ConvBlock(cin, self.widths[0], depth=2)
        self.pool = nn.AvgPool2d(2)
        self.encoders = nn.ModuleList([
            ConvBlock(self.widths[i - 1], self.widths[i], depth=2)
            for i in range(1, len(self.widths))
        ])
        rev = list(reversed(self.widths))
        self.decoders = nn.ModuleList([
            UpBlock(rev[i], rev[i + 1], rev[i + 1]) for i in range(len(rev) - 1)
        ])
        self.head = nn.Conv2d(self.widths[0], n_outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.inc(x)]
        for enc in self.encoders:
            feats.append(enc(self.pool(feats[-1])))
        y = feats[-1]
        for i, dec in enumerate(self.decoders):
            y = dec(y, feats[-2 - i])
        return self.head(y)


class UNet2D(nn.Module):
    """B1: single central slice, four channels (or the available subset)."""

    def __init__(self, cfg: Optional[ModelConfig] = None, n_modalities: int = 4,
                 context_slices: int = 5, n_outputs: int = 3):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.center = context_slices // 2
        self.core = _UNet2DCore(n_modalities, cfg.widths, n_outputs)

    def forward(self, image, availability, quality, return_aux: bool = False):
        x = image[:, :, self.center] * availability[:, :, None, None]
        logits = self.core(x)
        return (logits, {"deep_logits": []}) if return_aux else logits


class UNet25D(nn.Module):
    """B2: five adjacent slices x four modalities as channels + availability mask.

    The dimensionality-matched control: same 2.5D context as QMMF-Net, but the
    modality handling is plain channel concatenation.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, n_modalities: int = 4,
                 context_slices: int = 5, n_outputs: int = 3):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.n_modalities = n_modalities
        self.context_slices = context_slices
        # +M channels carry the availability mask explicitly, so this baseline
        # is not handicapped by having to infer which sequences are missing.
        cin = n_modalities * context_slices + n_modalities
        self.core = _UNet2DCore(cin, cfg.widths, n_outputs)

    def forward(self, image, availability, quality, return_aux: bool = False):
        b, m, d, h, w = image.shape
        x = (image * availability[:, :, None, None, None]).reshape(b, m * d, h, w)
        mask_planes = availability[:, :, None, None].expand(b, m, h, w)
        logits = self.core(torch.cat([x, mask_planes], dim=1))
        return (logits, {"deep_logits": []}) if return_aux else logits


class HeMIS25D(nn.Module):
    """B3: shared modality stem + equal-weight masked mean and variance.

    The foundational arbitrary-modality baseline (Havaei et al., MICCAI 2016),
    reimplemented at 2.5D under this study's pipeline. Ablation A3 is the same
    fusion inside QMMF-Net's body; this is the standalone architecture.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, n_modalities: int = 4,
                 context_slices: int = 5, n_outputs: int = 3):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.widths = tuple(cfg.widths)
        self.stem = PseudoConv3DStem(context_slices, self.widths[0],
                                     cond_dim=cfg.modality_embed_dim, use_film=False)
        self.fuse = EqualMaskedMean(self.widths[0], with_variance=True)
        self.core_encoders = nn.ModuleList([
            ConvBlock(self.widths[i - 1], self.widths[i], depth=2)
            for i in range(1, len(self.widths))
        ])
        self.pool = nn.AvgPool2d(2)
        rev = list(reversed(self.widths))
        self.decoders = nn.ModuleList([
            UpBlock(rev[i], rev[i + 1], rev[i + 1]) for i in range(len(rev) - 1)
        ])
        self.head = nn.Conv2d(self.widths[0], n_outputs, 1)

    def forward(self, image, availability, quality, return_aux: bool = False):
        b, m, d, h, w = image.shape
        feats = self.stem(image.reshape(b * m, d, h, w), None)
        feats = feats.reshape(b, m, *feats.shape[1:])
        x, aux = self.fuse(feats, availability, quality, None)
        skips = [x]
        for enc in self.core_encoders:
            skips.append(enc(self.pool(skips[-1])))
        y = skips[-1]
        for i, dec in enumerate(self.decoders):
            y = dec(y, skips[-2 - i])
        logits = self.head(y)
        if return_aux:
            return logits, {"deep_logits": [], "gate_alpha": aux["alpha"].unsqueeze(1)}
        return logits


class SegResNet3DWrapper(nn.Module):
    """B4: small 3D SegResNet (MONAI) on cropped patches.

    Wrapped so it accepts the same call signature. It consumes the 2.5D window
    as a short 3D volume; the plan treats it as the full-modality volumetric
    reference, and its comparison is the H1 non-inferiority test.

    Requires MONAI. Raises a clear error rather than silently substituting a
    different architecture.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, n_modalities: int = 4,
                 context_slices: int = 5, n_outputs: int = 3, init_filters: int = 16):
        super().__init__()
        try:
            from monai.networks.nets import SegResNet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Baseline B4 needs MONAI: pip install monai. "
                "Do not substitute another 3D network silently."
            ) from exc
        self.n_modalities = n_modalities
        self.net = SegResNet(
            spatial_dims=3, init_filters=init_filters,
            in_channels=n_modalities, out_channels=n_outputs,
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
        )
        self.center = context_slices // 2

    def forward(self, image, availability, quality, return_aux: bool = False):
        # image [B, M, D, H, W] is already the right layout for a 3D network.
        x = image * availability[:, :, None, None, None]
        out = self.net(x)                       # [B, 3, D, H, W]
        logits = out[:, :, self.center]         # supervise the central slice
        return (logits, {"deep_logits": []}) if return_aux else logits


class SwinUNETRWrapper(nn.Module):
    """B6 (optional): tiny SwinUNETR, included only if it clears the pilot gate.

    Plan 11.5: if this OOMs or runs >1.5x QMMF-Net runtime, it is removed from
    the mandatory comparisons and the feasibility failure is documented rather
    than hidden.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, n_modalities: int = 4,
                 context_slices: int = 5, n_outputs: int = 3,
                 img_size=(64, 192, 192), feature_size: int = 24):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Baseline B6 needs MONAI: pip install monai.") from exc
        self.net = SwinUNETR(
            img_size=img_size, in_channels=n_modalities,
            out_channels=n_outputs, feature_size=feature_size,
        )
        self.center = context_slices // 2

    def forward(self, image, availability, quality, return_aux: bool = False):
        x = image * availability[:, :, None, None, None]
        logits = self.net(x)[:, :, self.center]
        return (logits, {"deep_logits": []}) if return_aux else logits
