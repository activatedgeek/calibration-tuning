import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


class AnyCosineScheduler:
    """Hacky repurpose of torch schedulers for all scalars (any non-nn.Parameter)."""

    def __init__(self, *args, **kwargs) -> None:
        self.setup(*args, **kwargs)

    def setup(self, init_value=None, T_max=None, last_epoch=-1, **kwargs):
        if init_value is None:
            return

        self._p = nn.Parameter(torch.randn(1))
        self._T_max = T_max
        self._opt = torch.optim.SGD([self._p], lr=init_value)
        self._sched = CosineAnnealingLR(self._opt, T_max, **kwargs)
        if last_epoch > 0:
            for _ in range(last_epoch):
                self.step()

    @property
    def value(self):
        return self._sched.get_last_lr()[0]

    def step(self):
        if self._T_max > 0:
            self._sched.step()
