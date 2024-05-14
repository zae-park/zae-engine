import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict, Union, Optional, Iterable

import tqdm
import wandb
import numpy as np
import torch
from torch import optim
from torch.utils import data as td

from .add_on import NeptuneLogger
from ..schedulers import core


class Trainer(ABC):
    """
    Abstract class for experiments with 2 abstract methods train_step and test_step.
    Note that both the train_step and test_step are receive batch and return a dictionary.
    More detail about those are in each method's docstring.

    :param model: nn.Module
    :param device: torch.device
    :param mode: str
    :param optimizer: str, optional
    :param scheduler: str, optional
    :callbacks: Union[Callable, Iterable], optional
    :callback_step: int, optional
    :callback_epoch
    """

    def __init__(
        self,
        model,
        device: torch.device,
        mode: str,
        optimizer: optim.Optimizer,
        scheduler: Optional[Union[optim.lr_scheduler.LRScheduler, core.SchedulerBase]],
        log_bar: bool = True,
        callbacks: Iterable = (),
        web_logger: Optional[dict[str, Union[object, dict]]] = None,
        scheduler_step_on_batch: bool = False,
    ):
        # Init with given args
        if "cuda" in device.type:
            torch.cuda.set_device(device)  # Not for device in ['cpu', 'mps']
        self.device = device
        self.model = self._to_device(model)
        self.mode = mode
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_bar = log_bar
        self.progress_checker = ProgressChecker()

        # Init with options
        self.callbacks = callbacks
        self.scheduler_step_on_batch = scheduler_step_on_batch

        # Init vars
        self.log_train, self.log_test = defaultdict(list), defaultdict(list)
        self.loss_buffer, self.weight_buffer = torch.inf, defaultdict(list)
        self.loader, self.n_data, self.batch_size = None, None, None
        self.valid_loader, self.n_valid_data, self.valid_batch_size = None, None, None

        if web_logger:
            self.web_logger = self.check_web_logger(web_logger)

    def _to_cpu(self, *args) -> Tuple[torch.Tensor, ...] or torch.Tensor:
        """
        Cast given arguments to cpu.
        :param args: Single argument or variable-length sequence of arguments.
        :return: Single argument or variable-length sequence of arguments in cpu.
        """
        if len(args) == 1:
            a = args[0]
            if isinstance(a, torch.Tensor):
                return a.detach().cpu() if "detach" in a.__dir__() else a.item() if "item" in a.__dir__() else a
            else:
                return a
        else:
            return tuple([self._to_cpu(a) for a in args])

    def _to_device(
        self, *args, **kwargs
    ) -> Tuple[Union[torch.Tensor, torch.nn.Module]] or Union[torch.Tensor, torch.nn.Module]:
        """
        Cast given arguments to device.
        :param args: Single argument or variable-length sequence of arguments.
        :return: Single argument or variable-length sequence of arguments in device.
        """
        if args:
            if len(args) == 1:
                return args[0].to(self.device) if "to" in args[0].__dir__() else args[0]
            else:
                return tuple([a.to(self.device) if "to" in a.__dir__() else a for a in args])
        elif kwargs:
            return {k: v.to(self.device) if "to" in v.__dir__() else v for k, v in kwargs.items()}

    def _data_count(self, initial=False) -> None:
        """
        Count the number of data in loader.
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
        """
        Check batch size of loader.
        """
        self.batch_size = self.loader.batch_size if self.loader is not None else 0
        self.valid_batch_size = self.valid_loader.batch_size if self.valid_loader is not None else 0

    def _scheduler_step_check(self, epoch: int) -> None:
        if "total_iters" in self.scheduler.__dict__ and self.scheduler_step_on_batch:
            remain = 1 if (not self.loader.drop_last) and (self.n_data % self.batch_size) else 0
            self.batch_cnt = self.n_data // self.batch_size + remain
            need_steps = epoch * self.batch_cnt
            assert self.scheduler.total_iters >= need_steps, (
                f'The "total_iters" {self.scheduler.total_iters} for the given scheduler is insufficient.'
                f"It must be at least more than the total iterations {need_steps} required during training."
            )

    def run(self, n_epoch: int, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **kwargs) -> None:
        """
        Run for a given loader.
        If valid_loader is not None, the model evaluates the data in valid_loader for every epoch.
        :param n_epoch:
        :param loader:
        :param valid_loader:
        :return:
        """
        self.loader, self.valid_loader = loader, valid_loader
        self._check_batch_size()
        self._scheduler_step_check(n_epoch)
        progress = tqdm.tqdm(range(n_epoch), position=0, leave=True) if self.log_bar else range(n_epoch)
        for e in progress:
            if self.log_bar:
                progress.set_description("Epoch %d" % (e + 1))
            else:
                print("Epoch %d" % (e + 1))
            self._data_count(initial=True)
            self.run_epoch(loader, **kwargs)
            if valid_loader:
                self.toggle()
                self.run_epoch(valid_loader, **kwargs)
                self.toggle()
            if self.mode == "train":
                cur_loss = np.mean(self.log_test["loss"] if valid_loader else self.log_train["loss"]).item()
                self.check_better(cur_epoch=e + 1, cur_loss=cur_loss)
                if not self.scheduler_step_on_batch:
                    self.scheduler.step(**kwargs)
                self.progress_checker.update_epoch()

    def run_callback(self):
        if self.callbacks:
            for cb in self.callbacks:
                cb(self)  # replace even

    def run_epoch(self, loader: td.DataLoader, **kwargs) -> None:
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
        Run for a batch (not epoch)
        """
        batch = self._to_device(**batch) if isinstance(batch, dict) else self._to_device(*batch)
        if self.mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            step_dict = self.train_step(batch)
            step_dict["loss"].backward()
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
        self.run_callback()

    @abstractmethod
    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        """
        Unused dummy function yet exist to provide I/O format information.
        The batch is a part of the dataset in the dataloader fetched via  __getitem__ method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function work only in 'train' mode, hence back() method in nn.Module is necessary.
        """
        x, y, fn = batch  # or x, y, fn = batch['x'], batch['y'] batch['fn']
        outputs = self.model(x)
        loss = [0.0] * len(x)
        return {"loss": loss, "mean": torch.mean(outputs)}

    @abstractmethod
    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        """
        Unused dummy function yet exist to provide I/O format information.
        The batch is a part of the dataset in the dataloader fetched via  __getitem__ method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function work only in 'test' mode.
        """
        x, y, fn = batch  # or x, y, fn = batch['x'], batch['y'] batch['fn']
        outputs = self.model(x)
        loss = [0.0] * len(x)
        return {"loss": loss, "mean": torch.mean(outputs)}

    def logging(self, step_dict: Dict[str, torch.Tensor]) -> None:
        for k, v in step_dict.items():
            if self.mode == "train":
                self.log_train[k].append(self._to_cpu(v))
            elif self.mode == "test":
                self.log_test[k].append(self._to_cpu(v))
            else:
                raise ValueError(f"Unexpected mode {self.mode}.")

    def toggle(self, mode: str = None) -> None:
        """
        Switching mode in instance.
        :param mode: str
        """
        if mode:
            self.mode = mode
        else:
            self.mode = "test" if self.mode == "train" else "train"

    def check_better(self, cur_epoch: int, cur_loss: float) -> None:
        """
        Compare the current model and the in-buffer model.
        The criterion is loss.
        :param cur_epoch: int
        :param cur_loss: float
        """
        if cur_loss > self.loss_buffer:
            return
        self.weight_buffer["epoch"].append(cur_epoch)
        self.weight_buffer["weight"].append(self.model.state_dict())

    def log_reset(self) -> None:
        """
        Clear the log.
        """
        self.log_train.clear()
        self.log_test.clear()

    def print_log(self, cur_batch: int, num_batch: int) -> Tuple[str, dict]:
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
        # print(log_str, end=end, file=disp)
        return log_str, {"end": end, "file": disp}

    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)

    def apply_weights(self, filename: str = None, strict: bool = True) -> None:
        """
        Update model's weights.
        If filename is None, model updated with latest weight buffer in the instance.
        If not, the model is updated with weights loaded from the filename.
        :param filename: str
        :param strict: bool
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

    def check_web_logger(self, web_logger: Dict[str, Union[object, dict]]) -> Dict[str, Union[object, None]]:
        """
        Check given web_logger is runner object or runner params.
        If runner params are provided, init new runner object using those.
        :param web_logger:
        :return:
        """
        logger_dict = defaultdict(None)
        for k, v in web_logger.items():
            if k.lower() in ["wandb", "w&b"]:
                logger_dict["wandb"] = wandb.init(**v) if isinstance(v, dict) else v
            elif k.lower() in ["neptune", "neptune.ai", "neptuneai", "neptune-ai", "nep"]:
                logger_dict["neptune"] = self.init_tkn(**v) if isinstance(v, dict) else v
            else:
                # zae-engine now supports WandB and Neptune.ai only.
                continue
        return logger_dict

    def init_tkn(self, project_name: str, api_tkn: str = "", **kwargs) -> None:
        """
        Initialize neptune logger with given project name and token.
        Usage:
            trainer.init_tkn(PROJECT_NAME, API_TOKEN)
        :param project_name: str
        :param api_tkn: str
        :param kwargs: extra hashable for tracking model (or opt, scheduler, etc...)
        :return: None
        """
        self.web_logger = NeptuneLogger(project_name, api_tkn, **kwargs)

    def end_web_logger(self) -> None:
        """
        Eliminate web_loggers.
        :return: None
        """
        if wb_log := self.web_logger["wandb"]:
            wb_log.finish()
        if np_log := self.web_logger["neptune"]:
            np_log.eliminate()

    def inference(self, loader) -> list:
        # if self.web_logger:
        #     self.end_web_logger()
        self.toggle("test")
        self.run(n_epoch=1, loader=loader)
        return self.log_test["output"]


class ProgressChecker:
    def __init__(self):
        self.__step = 1
        self.__epoch = 1

    def update_step(self):
        self.__step += 1

    def update_epoch(self):
        self.__epoch += 1

    def get_step(self):
        return self.__step

    def get_epoch(self):
        return self.__epoch

    def init_state(self):
        self.__step = 1
        self.__epoch = 1
