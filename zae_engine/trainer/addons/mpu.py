from typing import Type, Tuple, Optional, Union, Sequence

import torch
import torch.utils.data as td
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from .core import AddOnBase
from .._trainer import T


class MultiGPUAddon(AddOnBase):
    """
    Add-on for enabling multi-GPU support using Distributed Data Parallel (DDP) in the Trainer class.
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        """
        Applies the multi-GPU modifications to the base class.

        Parameters
        ----------
        base_cls : Type
            The base class to which the multi-GPU modifications will be applied.

        Returns
        -------
        Type
            The modified base class with multi-GPU support.
        """

        class MultiGPUTrainer(base_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rank = kwargs.pop("rank", 0)
                self.world_size = kwargs.pop("world_size", 1)
                self.setup_ddp(self.rank, self.world_size)
                self.device = torch.device(f"cuda:{self.rank}")
                self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

            def _set_device(self, device: Union[torch.device, Sequence[torch.device]]):
                # Device is set up already in __init__
                pass

            def _to_device(
                self, *args, **kwargs
            ) -> Tuple[Union[torch.Tensor, torch.nn.Module]] or Union[torch.Tensor, torch.nn.Module]:
                """
                Cast given arguments to device.

                Parameters
                ----------
                args : Single argument or variable-length sequence of arguments.
                    The arguments to be cast to the specified device.

                Returns
                -------
                Single argument or variable-length sequence of arguments in device.
                """
                if isinstance(self.device, torch.device):
                    if args:
                        if len(args) == 1:
                            return args[0].to(self.device) if "to" in args[0].__dir__() else args[0]
                        else:
                            return tuple([a.to(self.device) if "to" in a.__dir__() else a for a in args])
                    elif kwargs:
                        return {k: v.to(self.device) if "to" in v.__dir__() else v for k, v in kwargs.items()}
                else:
                    # Handle multiple GPUs
                    for d in self.device:
                        if args:
                            if len(args) == 1:
                                return args[0].to(d) if "to" in args[0].__dir__() else args[0]
                            else:
                                return tuple([a.to(d) if "to" in a.__dir__() else a for a in args])
                        elif kwargs:
                            return {k: v.to(d) if "to" in v.__dir__() else v for k, v in kwargs.items()}

            def setup_ddp(self, rank, world_size):
                dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
                torch.cuda.set_device(rank)

            def cleanup_ddp(self):
                dist.destroy_process_group()

            def run(
                self, n_epoch: int, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **kwargs
            ) -> None:
                """
                Run the training/testing process for a given number of epochs with DistributedSampler.

                Parameters
                ----------
                n_epoch : int
                    The number of epochs to run.
                loader : td.DataLoader
                    The data loader for the training/testing data.
                valid_loader : Optional[td.DataLoader], optional
                    The data loader for validation data, by default None.
                """
                # Assign DistributedSampler to loaders
                loader.sampler = DistributedSampler(loader.dataset, num_replicas=self.world_size, rank=self.rank)
                if valid_loader:
                    valid_loader.sampler = DistributedSampler(
                        valid_loader.dataset, num_replicas=self.world_size, rank=self.rank
                    )

                super().run(n_epoch, loader, valid_loader, **kwargs)

            def __del__(self):
                self.cleanup_ddp()

        return MultiGPUTrainer
