from functools import partial
from collections import defaultdict
from typing import Dict, Union

import torch
import wandb
import neptune as neptune

from .core import AddOnBase, T


class WandBLoggerAddon(AddOnBase):
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


# class NeptuneLogger:
#     def __init__(self, project_name: str, api_tkn: str = "", **kwargs):
#         """
#         Initial neptune tokens using given token.
#
#         :param project_name: Logging path to let neptune track each experiment.
#         :param api_tkn: Authorization token
#         :param kwargs: Auxiliary arguments to track models.
#         :return:
#         """
#         self.api_root = f"zae-park/{project_name}"
#         self.api_tkn = api_tkn
#         self.kwargs = kwargs
#
#         self.run, self.model = None, None
#         self.init()
#
#     def default(self):
#         self.run, self.model = None, None
#
#     def init(self):
#         # name = self.kwargs['name'] if 'name' in self.kwargs.keys() else 'Prediction model'
#         # key = self.kwargs['key'] if 'key' in self.kwargs.keys() else 'MOD'
#
#         if self.api_tkn:
#             try:
#                 self.run = nep.init_run(project=self.api_root, api_token=self.api_tkn)
#                 # self.model = nep.init_model(name=name, key=key, project=self.api_root, api_token=self.api_tkn)
#             except InvalidTkn as e:
#                 print(f'{"-" * 100}')
#                 print(
#                     "Receive invalid api token. Fail to generate Neptune instance, "
#                     "please check again @ https://app.neptune.ai/o/zae-park/-/projects"
#                 )
#                 print(f'{"-" * 100}')
#                 raise e
#             else:
#                 self.add_state_checker(self.run)
#
#     @staticmethod
#     def add_state_checker(*objects):
#         for obj in objects:
#             obj.is_live = partial(lambda self: self._state.value != "stopped", obj)
#
#     def log(self, key, value):
#         if self.run.is_live():
#             self.run[key].log(value)
#
#     def eliminate(self):
#         if isinstance(self.run, nep.metadata_containers.MetadataContainer):
#             self.run.stop()
#         if isinstance(self.model, nep.metadata_containers.MetadataContainer):
#             self.model.stop()
