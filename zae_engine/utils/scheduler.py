import math
from abc import ABC, abstractmethod

from torch.optim import lr_scheduler, Optimizer


class SchedulerBase(lr_scheduler.LRScheduler, ABC):
    def __init__(self, optimizer: Optimizer, total_steps, eta_min, last_epoch: int = -1):
        """
        optimizer: Adam, AdamW, ...
        total_steps: # of steps.
        eta_max: Minimum value of learning rate.
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._step_count = 0
        # super(SchedulerBase, self).__init__(optimizer, last_epoch)
        lr_scheduler.LRScheduler.__init__(self, optimizer=optimizer, last_epoch=last_epoch)

    @abstractmethod
    def get_lr(self):
        return 1.0


class SchedulerChain(SchedulerBase):
    def __init__(self, *schedulers: SchedulerBase):
        self.schedulers = schedulers
        self.steps = self.step_check()
        self.next_steps = self.steps[:-1]
        self.total_steps = self.steps[-1]

        self.i_scheduler = 0
        tmp = self.initializer()
        lr_scheduler.LRScheduler.__init__(self, optimizer=tmp[0], last_epoch=tmp[-1])
        # super(SchedulerChain).__init__(*self.initializer())

    def initializer(self):
        schedule = self.schedulers[self.i_scheduler]
        return schedule.optimizer, schedule.total_steps, schedule.eta_min, schedule.last_epoch

    def step_check(self):
        schedulers_steps = [scheduler.total_steps for scheduler in self.schedulers]
        return [sum(schedulers_steps[: i + 1]) for i in range(len(schedulers_steps))]

    def next_scheduler(self):
        self.i_scheduler += 1
        self.initializer()

    def step(self, epoch=None):
        if self._step_count in self.next_steps:
            self.next_scheduler()
        self.schedulers[self.i_scheduler].step(epoch)
        self._step_count += 1

    def get_lr(self):
        return self.schedulers[self.i_scheduler].get_lr()


class WarmUpScheduler(SchedulerBase):
    def __init__(self, optimizer: Optimizer, total_steps, eta_min, last_epoch: int = -1):
        super(WarmUpScheduler, self).__init__(optimizer, total_steps, eta_min, last_epoch)

    def get_lr(self):
        return [self.eta_min + (lr - self.eta_min) * self._step_count / self.total_steps for lr in self.base_lrs]


class CosineAnnealingScheduler(SchedulerBase):
    def __init__(self, optimizer: Optimizer, total_steps, eta_min, last_epoch: int = -1):
        super(CosineAnnealingScheduler, self).__init__(optimizer, total_steps, eta_min, last_epoch)

    def get_lr(self):
        lrs = []
        for lr in self.base_lrs:
            radian = math.pi * self._step_count / self.total_steps
            annealing = 0.5 * (math.cos(radian) + 1)
            offset = self.eta_min
            lrs.append(offset + (lr - offset) * annealing)
        return lrs
