import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict, Union, Optional, TypeVar, Type, Sequence

import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils import data as td
from torch.nn.parallel import DistributedDataParallel as DDP

# from .addons import AddOnBase
from ..schedulers import core

T = TypeVar("T", bound="Trainer")


class Trainer(ABC):
    """
    Abstract class for experiments with 2 abstract methods train_step and test_step.
    Note that both the train_step and test_step receive batch and return a dictionary.
    More detail about those are in each method's docstring.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained or tested.
    device : Union[torch.device, Sequence[torch.device]]
        The device(s) to run the model on (e.g., 'cuda' or ['cuda:0', 'cuda:1']).
    mode : str
        The mode of the trainer, either 'train' or 'test'.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    scheduler : Optional[Union[torch.optim.lr_scheduler._LRScheduler, core.SchedulerBase]]
        The learning rate scheduler for the optimizer.
    log_bar : bool, optional
        Whether to display a progress bar during training/testing, by default True.
    scheduler_step_on_batch : bool, optional
        Whether to update the scheduler on each batch, by default False.
    gradient_clip : float, optional
        Gradient clipping value. If 0, no gradient clipping is applied, by default 0.0.
    """

    def __init__(
        self,
        model,
        device: Union[torch.device, Sequence[torch.device]],
        mode: str,
        optimizer: optim.Optimizer,
        scheduler: Optional[Union[optim.lr_scheduler.LRScheduler, core.T]],
        *,
        log_bar: bool = True,
        scheduler_step_on_batch: bool = False,
        gradient_clip: float = 0.0,
    ):
        self.primary_device_index = 0  # Default index is 0, will be adjusted in multi-GPU cases
        # Init with given args (positional params)
        self._set_device(device)
        self.model = self._to_device(model)
        self.mode = mode
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Init with given args (named params)
        self.log_bar = log_bar
        self.scheduler_step_on_batch = scheduler_step_on_batch
        self.gradient_clip = gradient_clip

        self.progress_checker = ProgressChecker()

        # Init vars
        self.log_train, self.log_test = defaultdict(list), defaultdict(list)
        self.loss_history_train, self.loss_history_test = [], []
        self.loss_buffer, self.weight_buffer = torch.inf, defaultdict(list)
        self.loader, self.n_data, self.batch_size = None, None, None
        self.valid_loader, self.n_valid_data, self.valid_batch_size = None, None, None

    @classmethod
    def add_on(cls, *add_on_cls: "Type[AddOnBase]") -> Type[T]:
        """
        Install one or more add-ons to the Trainer class.

        Parameters
        ----------
        add_on_cls : Type[AddOnBase]
            One or more add-on classes to install.

        Returns
        -------
        Type[Trainer]
            The modified Trainer class with the add-ons applied.

        Examples
        --------
        >>> trainer = Trainer.add_on(MultiGPUAddon, SomeOtherAddon)(model, [device1, device2], mode='train', scheduler=scheduler, optimizer=optimizer)
        """
        base_cls = cls
        for add_on in add_on_cls:
            base_cls = add_on.apply(base_cls)
        return base_cls

    def _to_cpu(self, *args) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Cast given arguments to CPU.

        Parameters
        ----------
        args : Single argument or variable-length sequence of arguments.
            The arguments to be cast to CPU.

        Returns
        -------
        Single argument or variable-length sequence of arguments in CPU.
        """
        if len(args) == 1:
            a = args[0]
            if isinstance(a, torch.Tensor):
                return a.detach().cpu() if "detach" in a.__dir__() else a.item() if "item" in a.__dir__() else a
            else:
                return a
        else:
            return tuple([self._to_cpu(a) for a in args])

    def _set_device(self, device: Union[torch.device, Sequence[torch.device]]):
        if isinstance(device, (list, tuple)):
            # multi GPU
            self.device = device
        else:
            # single GPU
            self.device = [device]
            if "cuda" in device.type:
                torch.cuda.set_device(device)

    def _to_device(
        self, *args, **kwargs
    ) -> Union[Tuple[Union[torch.Tensor, torch.nn.Module]], Union[torch.Tensor, torch.nn.Module]]:
        """
        Cast given arguments to the appropriate device.

        Parameters
        ----------
        args : Single argument or variable-length sequence of arguments.
            The arguments to be cast to the specified device.

        Returns
        -------
        Single argument or variable-length sequence of arguments in device.
        """
        device = self.device[self.primary_device_index]  # Use the primary device index to select the correct device

        if args:
            if len(args) == 1:
                return args[0].to(device) if "to" in args[0].__dir__() else args[0]
            else:
                return tuple([a.to(device) if "to" in a.__dir__() else a for a in args])
        elif kwargs:
            return {k: v.to(device) if "to" in v.__dir__() else v for k, v in kwargs.items()}

    def _data_count(self, initial=False) -> None:
        """
        Count the number of data in loader.

        Parameters
        ----------
        initial : bool, optional
            Whether to count the initial number of data in the loader, by default False.
        """
        if initial:
            self.n_data = self.loader.dataset.__len__() if self.loader is not None else 0
            self.n_valid_data = self.valid_loader.dataset.__len__() if self.valid_loader is not None else 0
        else:
            if self.mode == "test" and self.valid_loader is not None:
                self.n_valid_data -= self.valid_batch_size
            else:
                self.n_data -= self.batch_size

    def _check_batch_size(self):
        """Check batch size of loader."""
        self.batch_size = self.loader.batch_size if self.loader is not None else 0
        self.valid_batch_size = self.valid_loader.batch_size if self.valid_loader is not None else 0

    def _scheduler_step_check(self, epoch: int) -> None:
        """
        Check if the scheduler's total iterations are sufficient for the given epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        if "total_iters" in self.scheduler.__dict__ and self.scheduler_step_on_batch:
            need_steps = epoch * len(self.loader)
            assert self.scheduler.total_iters >= need_steps, (
                f'The "total_iters" {self.scheduler.total_iters} for the given scheduler is insufficient.'
                f"It must be at least more than the total iterations {need_steps} required during training."
            )

    def run(self, n_epoch: int, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **kwargs) -> None:
        """
        Run the training/testing process for a given number of epochs.

        Parameters
        ----------
        n_epoch : int
            The number of epochs to run.
        loader : td.DataLoader
            The data loader for the training/testing data.
        valid_loader : Optional[td.DataLoader], optional
            The data loader for validation data, by default None.
        """
        self.loader, self.valid_loader = loader, valid_loader
        self._check_batch_size()
        self._scheduler_step_check(n_epoch)
        pre_epoch = self.progress_checker.get_epoch()
        progress = tqdm.tqdm(range(n_epoch), position=0, leave=True) if self.log_bar else range(n_epoch)
        for e in progress:
            e += pre_epoch
            if self.log_bar:
                progress.set_description("Epoch %d" % e)
            else:
                print("Epoch %d" % e)
            self._data_count(initial=True)
            self.run_epoch(loader, **kwargs)
            if valid_loader:
                self.toggle()
                self.run_epoch(valid_loader, **kwargs)
                self.toggle()
            if self.mode == "train":
                cur_loss = np.mean(self.log_test["loss"] if valid_loader else self.log_train["loss"]).item()
                self.check_better(cur_epoch=e, cur_loss=cur_loss)
                if not self.scheduler_step_on_batch:
                    self.scheduler.step(**kwargs)
                self.progress_checker.update_epoch()

    def run_epoch(self, loader: td.DataLoader, **kwargs) -> None:
        """
        Run the training/testing process for one epoch.

        Parameters
        ----------
        loader : td.DataLoader
            The data loader for the training/testing data.
        """
        self.log_reset()
        progress = tqdm.tqdm(loader, position=1, leave=False) if self.log_bar else loader
        for i, batch in enumerate(progress):
            self.run_batch(batch, **kwargs)
            self._data_count()
            desc, printer = self.print_log(cur_batch=i + 1, num_batch=len(loader))
            if self.log_bar:
                progress.set_description(desc)
            else:
                print(desc, **printer)

    def run_batch(self, batch: Union[tuple, dict], **kwargs) -> None:
        """
        Run the training/testing process for one batch.

        Parameters
        ----------
        batch : Union[tuple, dict]
            A batch of data.
        """
        batch = self._to_device(**batch) if isinstance(batch, dict) else self._to_device(*batch)
        if self.mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            step_dict = self.train_step(batch)
            step_dict["loss"].backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            if self.scheduler_step_on_batch:
                self.scheduler.step(**kwargs)

        elif self.mode == "test":
            self.model.eval()
            with torch.no_grad():
                step_dict = self.test_step(batch)
        else:
            raise ValueError(f"Unexpected mode {self.mode}.")

        self.logging(step_dict)
        self.progress_checker.update_step()

    @abstractmethod
    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        """
        Perform a training step.

        This method must be implemented by subclasses.
        This unused dummy function exists to provide I/O format information.

        The batch is a part of the dataset in the dataloader fetched via the `__getitem__` method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function works only in 'train' mode, hence the `backward()` method in `nn.Module` is necessary.

        Parameters
        ----------
        batch : Union[tuple, dict]
            A batch of data.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the results of the training step, including the loss.

        Examples
        --------
        >>> x, y, fn = batch['x'], batch['y'], batch['fn']  # or x, y, fn = batch
        >>> outputs = self.model(x)
        >>> loss = criteria(outputs)
        >>> return {"loss": loss, "mean_output": torch.mean(outputs)}
        """

        raise NotImplementedError("train_step must be implemented by subclasses")

    @abstractmethod
    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        """
        Perform a testing step.

        This method must be implemented by subclasses.
        This unused dummy function exists to provide I/O format information.

        The batch is a part of the dataset in the dataloader fetched via the `__getitem__` method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function works only in 'test' mode, and the `backward()` method is not necessary.

        Parameters
        ----------
        batch : Union[tuple, dict]
            A batch of data.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the results of the testing step, including the loss.

        Examples
        --------
        >>> x, y, fn = batch['x'], batch['y'], batch['fn']  # or x, y, fn = batch
        >>> outputs = self.model(x)
        >>> loss = criteria(outputs)
        >>> return {"loss": loss, "mean_output": torch.mean(outputs)}
        """

        raise NotImplementedError("test_step must be implemented by subclasses")

    def logging(self, step_dict: Dict[str, torch.Tensor]) -> None:
        """
        Log the results of a step.

        Parameters
        ----------
        step_dict : Dict[str, torch.Tensor]
            A dictionary containing the results of a training or testing step.
        """
        for k, v in step_dict.items():
            v_ = self._to_cpu(v)
            if self.mode == "train":
                self.log_train[k].append(v_)
                self.loss_history_train.append(v_)
            elif self.mode == "test":
                self.log_test[k].append(v_)
                self.loss_history_test.append(v_)
            else:
                raise ValueError(f"Unexpected mode {self.mode}.")

    def toggle(self, mode: str = None) -> None:
        """
        Switch the mode of the trainer.

        Parameters
        ----------
        mode : str, optional
            The mode to switch to, by default None.
            If None, toggles between 'train' and 'test'.
        """
        if mode:
            self.mode = mode
        else:
            self.mode = "test" if self.mode == "train" else "train"

    def check_better(self, cur_epoch: int, cur_loss: float) -> bool:
        """
        Compare the current model and the in-buffer model to determine if the current model is better.
        The criterion is loss.

        Parameters
        ----------
        cur_epoch : int
            The current epoch number.
        cur_loss : float
            The current loss value.
        """
        if cur_loss >= self.loss_buffer:
            return False
        self.weight_buffer["epoch"].append(cur_epoch)
        self.weight_buffer["weight"].append(self.model.state_dict())
        self.loss_buffer = cur_loss
        return True

    def log_reset(self) -> None:
        """
        Clear the log.
        """
        self.log_train.clear()
        self.log_test.clear()

    def get_loss_history(self, mode: str = "train") -> list:
        """
        Retrieve the list of all step losses.

        Parameters
        ----------
        mode : str, optional
            The mode to retrieve losses for, either 'train' or 'test'. Default is 'train'.

        Returns
        -------
        list
            A list of loss values for each step.
        """
        if mode == "train":
            return self.loss_history_train
        elif mode == "test":
            return self.loss_history_test
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'.")

    def print_log(self, cur_batch: int, num_batch: int) -> Tuple[str, dict]:
        """
        Print the log for the current batch.

        Parameters
        ----------
        cur_batch : int
            The current batch number.
        num_batch : int
            The total number of batches.

        Returns
        -------
        Tuple[str, dict]
            A tuple containing the log string and additional printing options.
        """
        log = self.log_train if self.mode == "train" else self.log_test
        LR = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0
        is_end = cur_batch == num_batch
        disp = None if is_end else sys.stderr
        end = "\n" if is_end else ""

        log_str = f"\r\t\tBatch: {cur_batch}/{num_batch}"
        for k, v in log.items():
            if "output" in k:
                continue
            log_str += f"\t{k}: {np.mean(v):.6f}"
        log_str += f"\tLR: {LR:.3e}"
        return log_str, {"end": end, "file": disp}

    def save_model(self, filename: str) -> None:
        """
        Save the model's state dictionary to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the model's state dictionary.
        """
        torch.save(self.model.state_dict(), filename)

    def apply_weights(self, filename: str = None, strict: bool = True) -> None:
        """
        Update model's weights.
        If filename is None, model updated with latest weight buffer in the instance.
        If not, the model is updated with weights loaded from the filename.

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the weights from, by default None.
        strict : bool, optional
            Whether to strictly enforce that the keys in `state_dict` match the keys returned by `model.state_dict()`, by default True.
        """
        try:
            if filename:
                weights = torch.load(filename, map_location=torch.device("cpu"))
            else:
                weights = self.weight_buffer["weight"][-1]
            self.model.load_state_dict(weights, strict=strict)
        except FileNotFoundError as e:
            print(f"Cannot find file {filename}", e)
        except IndexError as e:
            print(f"There is no weight in buffer of trainer.", e)

    def inference(self, loader) -> list:
        """
        Run inference on the given data loader.

        Parameters
        ----------
        loader : td.DataLoader
            The data loader for inference.

        Returns
        -------
        list
            The inference results.
        """
        self.toggle("test")
        self.run(n_epoch=1, loader=loader)
        return self.log_test["output"]


class ProgressChecker:
    """
    A helper class to track the progress of training/testing.

    Attributes
    ----------
    __step : int
        The current step count.
    __epoch : int
        The current epoch count.
    """

    def __init__(self):
        self.__step = 1
        self.__epoch = 1

    def update_step(self):
        """Update the step count."""
        self.__step += 1

    def update_epoch(self):
        """Update the epoch count."""
        self.__epoch += 1

    def get_step(self):
        """
        Get the current step count.

        Returns
        -------
        int
            The current step count.
        """
        return self.__step

    def get_epoch(self):
        """
        Get the current epoch count.

        Returns
        -------
        int
            The current epoch count.
        """
        return self.__epoch

    def init_state(self):
        """Initialize the state by resetting step and epoch counts to 1."""
        self.__step = 1
        self.__epoch = 1
