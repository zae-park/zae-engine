import os
import torch
import pickle
from torch import nn, optim
from typing import Type

import safetensors.torch  # Ensure safetensors is installed

from .core import AddOnBase, T

class StateSaverAddon(AddOnBase):
    """
    Add-on for saving checkpoints of the model, optimizer, and scheduler during training.

    Methods
    -------
    apply(cls, base_cls)
        Applies the checkpoint saving modifications to the base class.
    """

    def __init__(self, checkpoint_dir: str, save_model_format: str = "ckpt"):
        self.checkpoint_dir = checkpoint_dir
        self.save_model_format = save_model_format

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        """
        Applies the checkpoint saving modifications to the base class.

        Parameters
        ----------
        base_cls : Type[T]
            The base class to which the checkpoint saving modifications will be applied.

        Returns
        -------
        Type[T]
            The modified base class with checkpoint saving support.
        """
        
        class StateSaverTrainer(base_cls):
            def __init__(self, *args, checkpoint_dir: str, save_model_format: str = "ckpt", **kwargs):
                super().__init__(*args, **kwargs)
                self.checkpoint_dir = checkpoint_dir
                self.save_model_format = save_model_format
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)

            def save_checkpoint(self, epoch: int) -> None:
                """
                Save the model, optimizer, and scheduler state to the checkpoint directory.

                Parameters
                ----------
                epoch : int
                    The current epoch number.
                """
                model_path = os.path.join(self.checkpoint_dir, f"model_{epoch}.{self.save_model_format}")
                optimizer_path = os.path.join(self.checkpoint_dir, "optimizer.zae")
                scheduler_path = os.path.join(self.checkpoint_dir, "scheduler.zae")

                if self.save_model_format == "ckpt":
                    torch.save(self.model.state_dict(), model_path)
                elif self.save_model_format == "safetensors":
                    
                    safetensors.torch.save_file(self.model.state_dict(), model_path)
                else:
                    raise ValueError("Unsupported model save format. Use 'ckpt' or 'safetensors'.")

                with open(optimizer_path, 'wb') as f:
                    pickle.dump(self.optimizer.state_dict(), f)

                with open(scheduler_path, 'wb') as f:
                    pickle.dump(self.scheduler.state_dict(), f)

            def run_epoch(self, loader: td.DataLoader, **kwargs) -> None:
                """
                Run the training/testing process for one epoch.

                Parameters
                ----------
                loader : td.DataLoader
                    The data loader for the training/testing data.
                """
                super().run_epoch(loader, **kwargs)
                self.save_checkpoint(self.progress_checker.get_epoch())

        return StateSaverTrainer