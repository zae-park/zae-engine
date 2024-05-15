import math

from torch.optim import Optimizer
from . import core


class WarmUpScheduler(core.SchedulerBase):
    def __init__(self, optimizer: Optimizer, total_iters, eta_min: float = 0, last_epoch: int = -1):
        super(WarmUpScheduler, self).__init__(optimizer, total_iters, eta_min, last_epoch)

    def get_lr(self):
        return [self.eta_min + (lr - self.eta_min) * self._step_count / self.total_iters for lr in self.base_lrs]


class CosineAnnealingScheduler(core.SchedulerBase):
    def __init__(self, optimizer: Optimizer, total_iters, eta_min: float = 0, last_epoch: int = -1):
        super(CosineAnnealingScheduler, self).__init__(optimizer, total_iters, eta_min, last_epoch)

    def get_lr(self):
        lrs = []
        for lr in self.base_lrs:
            radian = math.pi * self._step_count / self.total_iters
            annealing = 0.5 * (math.cos(radian) + 1)
            offset = self.eta_min
            lrs.append(offset + (lr - offset) * annealing)
        return lrs
