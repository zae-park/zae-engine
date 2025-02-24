from functools import partial
from collections import defaultdict
from typing import Dict, Union

import torch
import wandb
import neptune as neptune

from .core import AddOnBase, T
from ..addons import ADDON_CASE_DEPENDENT


class WandBLoggerAddon(AddOnBase):
    addon_case = ADDON_CASE_DEPENDENT  # Needs data of previous Add-on
    """
    Add-on for real-time logging with Weights & Biases (WandB).

    This add-on integrates WandB into the training process, allowing users to log
    metrics and monitor training progress in real-time.

    Parameters
    ----------
    web_logger : dict, optional
        Configuration dictionary for initializing WandB. Must include a key 'wandb'
        with WandB initialization parameters.

    Methods
    -------
    logging(step_dict: Dict[str, torch.Tensor])
        Log metrics to WandB during each step.
    init_wandb(params: dict)
        Initialize WandB with the given parameters.

    Notes
    -----
    This add-on requires WandB to be installed and a valid API key to be available.

    Examples
    --------
    Using WandBLoggerAddon for real-time logging:

    >>> from zae_engine.trainer import Trainer
    >>> from zae_engine.trainer.addons import WandBLoggerAddon

    >>> MyTrainer = Trainer.add_on(WandBLoggerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     web_logger={"wandb": {"project": "my_project"}}
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)
    """

    @classmethod
    def apply(cls, base_cls: T) -> T:
        class WandBLogger(base_cls):
            def __init__(self, *args, **kwargs):
                web_logger = kwargs.pop("web_logger", None)
                super().__init__(*args, **kwargs)
                if web_logger and "wandb" in web_logger:
                    self.web_logger = self.init_wandb(web_logger["wandb"])

            def init_wandb(self, params: dict):
                return wandb.init(**params)

            def logging(self, step_dict: Dict[str, torch.Tensor]) -> None:
                super().logging(step_dict)
                if hasattr(self, "web_logger"):
                    wandb.log({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in step_dict.items()})

            def __del__(self):
                if hasattr(self, "web_logger"):
                    self.web_logger.finish()

        return WandBLogger


class NeptuneLoggerAddon(AddOnBase):
    addon_case = ADDON_CASE_DEPENDENT  # Needs data of previous Add-on
    """
    Add-on for real-time logging with Neptune.

    This add-on integrates Neptune into the training process, enabling real-time logging
    of metrics and other training details. It also provides functionality to monitor and
    track experiments remotely.

    Parameters
    ----------
    web_logger : dict, optional
        Configuration dictionary for initializing Neptune. Must include a key 'neptune'
        with Neptune initialization parameters, such as 'project_name' and 'api_tkn'.

    Methods
    -------
    logging(step_dict: Dict[str, torch.Tensor])
        Log metrics to Neptune during each step.
    init_neptune(params: dict)
        Initialize a Neptune run with the given parameters.

    Notes
    -----
    This add-on requires Neptune to be installed and a valid API token to be available.
    Ensure your Neptune project is properly set up to track experiments.

    Examples
    --------
    Using NeptuneLoggerAddon for real-time logging:

    >>> from zae_engine.trainer import Trainer
    >>> from zae_engine.trainer.addons import NeptuneLoggerAddon

    >>> MyTrainer = Trainer.add_on(NeptuneLoggerAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     web_logger={"neptune": {"project_name": "my_workspace/my_project", "api_tkn": "your_api_token"}}
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)

    Adding multiple loggers, including Neptune:

    >>> from zae_engine.trainer.addons import WandBLoggerAddon

    >>> MyTrainerWithLoggers = Trainer.add_on(WandBLoggerAddon, NeptuneLoggerAddon)
    >>> trainer_with_loggers = MyTrainerWithLoggers(
    >>>     model=my_model,
    >>>     device='cuda',
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler,
    >>>     web_logger={
    >>>         "wandb": {"project": "my_wandb_project"},
    >>>         "neptune": {"project_name": "my_workspace/my_neptune_project", "api_tkn": "your_api_token"}
    >>>     }
    >>> )
    >>> trainer_with_loggers.run(n_epoch=10, loader=train_loader)
    """

    @classmethod
    def apply(cls, base_cls: T) -> T:
        class NeptuneLogger(base_cls):
            def __init__(self, *args, **kwargs):
                web_logger = kwargs.pop("web_logger", None)
                super().__init__(*args, **kwargs)
                if web_logger and "neptune" in web_logger:
                    self.web_logger = self.init_neptune(web_logger["neptune"])

            def init_neptune(self, params: dict):
                project_name = params.pop("project_name", "")
                api_tkn = params.pop("api_tkn", "")
                run = neptune.init_run(project=project_name, api_token=api_tkn)
                self.add_state_checker(run)
                return run

            def logging(self, step_dict: Dict[str, torch.Tensor]) -> None:
                super().logging(step_dict)
                if hasattr(self, "web_logger"):
                    for k, v in step_dict.items():
                        self.web_logger[k].log(v.item() if isinstance(v, torch.Tensor) else v)

            def __del__(self):
                if hasattr(self, "web_logger"):
                    self.web_logger.stop()

            @staticmethod
            def add_state_checker(*objects):
                for obj in objects:
                    obj.is_live = partial(lambda self: self._state.value != "stopped", obj)

        return NeptuneLogger
