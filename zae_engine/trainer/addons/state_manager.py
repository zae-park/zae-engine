import os
import pickle
import torch
import safetensors.torch
from typing import Type, Union, Optional, Dict

from .core import AddOnBase, T


class StateManagerAddon(AddOnBase):
    def __init__(self, save_path: str, save_format: str = "ckpt"):
        self.save_path = save_path
        self.save_format = save_format

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        class TrainerWithStateManager(base_cls):
            def __init__(self, *args, **kwargs):
                self.save_path = kwargs.pop("save_path")
                self.save_format = kwargs.pop("save_format", "ckpt")
                super().__init__(*args, **kwargs)
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)

            def check_better(self, cur_epoch: int, cur_loss: float) -> bool:
                is_better = super().check_better(cur_epoch, cur_loss)
                if is_better:
                    self.save_state()
                return is_better

            def save_model(self, filename: str) -> None:
                if self.save_format == "ckpt":
                    torch.save(self.model.state_dict(), filename)
                elif self.save_format == "safetensor":
                    safetensors.torch.save_file(self.model.state_dict(), filename)

            def save_optimizer(self) -> None:
                with open(os.path.join(self.save_path, "optimizer.zae"), "wb") as f:
                    pickle.dump(self.optimizer.state_dict(), f)

            def save_scheduler(self) -> None:
                with open(os.path.join(self.save_path, "scheduler.zae"), "wb") as f:
                    pickle.dump(self.scheduler.state_dict(), f)

            def save_result(self) -> None:
                # TODO: Save result for validation set (or test)
                pass

            def save_state(self) -> None:
                """
                Save the state of the model, optimizer, and scheduler.
                """
                self.save_model(os.path.join(self.save_path, f"model.{self.save_format}"))
                self.save_optimizer()
                self.save_scheduler()

            def load_state(self) -> None:
                """
                Load the state of the model, optimizer, and scheduler.
                """
                model_file = os.path.join(self.save_path, f"model.{self.save_format}")
                optimizer_file = os.path.join(self.save_path, "optimizer.zae")
                scheduler_file = os.path.join(self.save_path, "scheduler.zae")

                if self.save_format == "ckpt" and os.path.exists(model_file):
                    self.model.load_state_dict(torch.load(model_file))
                elif self.save_format == "safetensor" and os.path.exists(model_file):
                    self.model.load_state_dict(safetensors.torch.load_file(model_file))

                if os.path.exists(optimizer_file):
                    with open(optimizer_file, "rb") as f:
                        self.optimizer.load_state_dict(pickle.load(f))

                if os.path.exists(scheduler_file):
                    with open(scheduler_file, "rb") as f:
                        self.scheduler.load_state_dict(pickle.load(f))

        return TrainerWithStateManager
