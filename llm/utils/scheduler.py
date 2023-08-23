import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


class AnyCosineScheduler:
    """Repurpose torch schedulers for all scalars (any non-nn.Parameter)."""

    def __init__(self, init_value, T_max, last_epoch=-1, **kwargs) -> None:
        p = nn.Parameter(torch.randn(1))
        self._T_max = T_max
        self._opt = torch.optim.SGD([p], lr=init_value)
        self._sched = CosineAnnealingLR(
            self._opt, T_max, last_epoch=last_epoch, **kwargs
        )

    @property
    def value(self):
        return self._sched.get_lr()[0]

    def step(self):
        if self._T_max > 0:
            self._sched.step()
