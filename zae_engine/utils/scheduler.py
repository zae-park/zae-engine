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


class CosineAnnealingWarmUpRestarts(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, T_up=0, eta_max=0.1, gamma=1.0, last_epoch=-1):
        """
        optimizer: Adam, AdamW, ...
        T_0: # of epochs in a cycle.
        T_multi: # of cycle. (T_0 * T_multi = total epochs)
        T_up: # of epochs for warm-up.
        eta_max: Maximum value of learning rate.
        gamma: Attenuation factor of learning rate.
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [self.eta_max * ((float(self.T_cur) + 1) / self.T_up) for _ in self.base_lrs]
        else:
            return [
                self.eta_max * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
