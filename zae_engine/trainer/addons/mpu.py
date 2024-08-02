from typing import Type, Optional

import torch
import torch.utils.data as td
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from .core import AddOnBase
from .._trainer import T


class MultiGPUAddon(AddOnBase):
    """
    Add-on for enabling multi-GPU support using Distributed Data Parallel (DDP) in the Trainer class.

    Methods
    -------
    apply(cls, base_cls)
        Applies the multi-GPU modifications to the base class.
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
                self.rank = kwargs.pop("rank", 0)
                self.world_size = kwargs.pop("world_size", torch.cuda.device_count())
                self.setup_ddp(self.rank, self.world_size)
                super().__init__(*args, **kwargs)
                self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
                self.device = torch.device(f"cuda:{self.rank}")

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
                loader.sampler = DistributedSampler(loader.dataset, num_replicas=self.world_size, rank=self.rank)
                if valid_loader:
                    valid_loader.sampler = DistributedSampler(
                        valid_loader.dataset, num_replicas=self.world_size, rank=self.rank
                    )

                super().run(n_epoch, loader, valid_loader, **kwargs)

            def __del__(self):
                self.cleanup_ddp()

        return MultiGPUTrainer
