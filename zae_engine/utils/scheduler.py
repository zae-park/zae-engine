import math

from torch.optim import lr_scheduler, Optimizer


class WarmUpScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps, eta_min, last_epoch: int = -1):
        """
        optimizer: Adam, AdamW, ...
        total_steps: # of steps.
        eta_max: Minimum value of learning rate.
        """

        self.total_steps = total_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._step_count = 0
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (lr - self.eta_min) * self._step_count / self.total_steps for lr in self.base_lrs]


class CosineAnnealingScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps, eta_min, last_epoch: int = -1):
        """
        optimizer: Adam, AdamW, ...
        total_steps: # of steps.
        eta_max: Minimum value of learning rate.
        """

        self.total_steps = total_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._step_count = 0
        super(CosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

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
