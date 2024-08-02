from typing import TypeVar
from abc import ABC, abstractmethod

from torch.optim import lr_scheduler, Optimizer


T = TypeVar("T", bound="SchedulerBase")


class SchedulerBase(lr_scheduler.LRScheduler, ABC):
    """
    Base class for learning rate schedulers.

    This class extends PyTorch's LRScheduler and adds additional functionality
    for custom learning rate scheduling.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer for which to schedule the learning rate.
    total_iters : int
        The total number of iterations for the scheduler.
    eta_min : float
        The minimum learning rate.
    last_epoch : int, optional
        The index of the last epoch. Default is -1.

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer being used.
    total_iters : int
        The total number of iterations for the scheduler.
    eta_min : float
        The minimum learning rate.
    last_epoch : int
        The index of the last epoch.
    _step_count : int
        The step count for the scheduler.
    """

    def __init__(self, optimizer: Optimizer, total_iters: int, eta_min: float, last_epoch: int = -1):
        """
        optimizer: Adam, AdamW, ...
        total_steps: # of steps.
        eta_max: Minimum value of learning rate.
        """
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._step_count = 0
        # super(SchedulerBase, self).__init__(optimizer, last_epoch)
        lr_scheduler.LRScheduler.__init__(self, optimizer=optimizer, last_epoch=last_epoch)

    @abstractmethod
    def get_lr(self):
        """
        Get the learning rate for the current epoch.

        Returns
        -------
        float
            The learning rate for the current epoch.
        """
        return 1.0


class SchedulerChain(SchedulerBase):
    """
    Chain multiple learning rate schedulers together.

    This class allows you to chain multiple learning rate schedulers so that
    they are applied sequentially.

    Parameters
    ----------
    *schedulers : SchedulerBase
        The schedulers to chain together.

    Attributes
    ----------
    schedulers : list of SchedulerBase
        The list of chained schedulers.
    optimizer : Optimizer
        The optimizer being used.
    next_iters : list of int
        The iteration counts at which to switch to the next scheduler.
    total_iters : int
        The total number of iterations for the entire chain of schedulers.
    i_scheduler : int
        The index of the current scheduler in the chain.
    """

    def __init__(self, *schedulers: SchedulerBase):
        self.schedulers = schedulers
        self.optimizer = self.sanity_check()

        iters = [s.total_iters for s in self.schedulers]
        accumulate_iters = [sum(iters[: i + 1]) for i in range(len(iters))]
        self.next_iters, self.total_iters = accumulate_iters[:-1], accumulate_iters[-1]

        self.i_scheduler = 0
        lr_scheduler.LRScheduler.__init__(self, optimizer=self.optimizer, last_epoch=-1)

    def sanity_check(self):
        """
        Check that all schedulers use the same optimizer.

        Returns
        -------
        Optimizer
            The common optimizer used by all schedulers.

        Raises
        ------
        AssertionError
            If multiple optimizers are detected.
        """
        # Check given schedulers for same optimizer
        opts = [s.optimizer for s in self.schedulers]
        cnt = len(set([id(o) for o in opts]))
        assert cnt == 1, (
            f"The given schedulers were expected to use single optimizer, "
            f"but {cnt} optimizers were detected: {opts}"
        )
        return opts[0]

    def step(self, epoch=None):
        """
        Perform a step of the scheduler.

        Parameters
        ----------
        epoch : int, optional
            The current epoch. If None, use the internal step count.
        """
        if self._step_count in self.next_iters:
            self.i_scheduler += 1
        self.schedulers[self.i_scheduler].step(epoch)
        self._step_count += 1

    def get_lr(self):
        """
        Get the learning rate for the current epoch.

        Returns
        -------
        float
            The learning rate for the current epoch.
        """
        return self.schedulers[self.i_scheduler].get_lr()
