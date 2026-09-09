"""Exponential moving average teacher (plan 8.4).

The teacher always sees the full four-modality input; the student sees a
sampled subset. No separate heavyweight teacher network is trained, which is
what keeps the consistency term affordable on a single T4.
"""

from __future__ import annotations

import copy
from typing import Iterator, Optional

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay_start: float = 0.99,
                 decay_end: float = 0.999, warmup_steps: int = 1000,
                 device: Optional[torch.device] = None):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.warmup_steps = max(warmup_steps, 1)
        self.step_count = 0

    def current_decay(self) -> float:
        """Ramp from decay_start to decay_end over warmup_steps."""
        t = min(self.step_count / self.warmup_steps, 1.0)
        return self.decay_start + (self.decay_end - self.decay_start) * t

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        msd = model.state_dict()
        for key, value in self.module.state_dict().items():
            source = msd[key]
            if value.dtype.is_floating_point:
                value.mul_(d).add_(source.detach().to(value.dtype), alpha=1.0 - d)
            else:
                value.copy_(source)
        self.step_count += 1

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self):
        return {"module": self.module.state_dict(), "step_count": self.step_count}

    def load_state_dict(self, state):
        self.module.load_state_dict(state["module"])
        self.step_count = int(state.get("step_count", 0))
