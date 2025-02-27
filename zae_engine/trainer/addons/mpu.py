import os
from typing import Type, Optional

# from ._mpu_core import MultiGPUTrainer
import torch
import torch.utils.data as td
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .core import AddOnBase
from .._trainer import T


class MultiGPUAddon(AddOnBase):
    """
    Add-on for distributed multi-GPU training.

    This add-on enables distributed training across multiple GPUs using PyTorch's
    Distributed Data Parallel (DDP). It handles process initialization, data distribution,
    and model synchronization for efficient multi-GPU training.

    Parameters
    ----------
    init_method : str, optional
        Initialization method for the distributed process group, typically a URL in the format
        `tcp://hostname:port`. Default is 'tcp://localhost:12355'.

    Methods
    -------
    run(n_epoch, loader, valid_loader=None, **aux_run_kwargs)
        Run the distributed training or testing process across multiple GPUs.
    train_process(rank, device_list, init_method, n_epoch, loader, valid_loader, aux_run_kwargs)
        Train the model on a specific GPU in the distributed setup.

    Notes
    -----
    This add-on requires multiple GPUs to be available and properly configured.

    Examples
    --------
    Using MultiGPUAddon for distributed training:

    >>> from zae_engine.trainer import Trainer
    >>> from zae_engine.trainer.addons import MultiGPUAddon

    >>> MyTrainer = Trainer.add_on(MultiGPUAddon)
    >>> trainer = MyTrainer(
    >>>     model=my_model,
    >>>     device=[torch.device('cuda:0'), torch.device('cuda:1')],
    >>>     optimizer=my_optimizer,
    >>>     scheduler=my_scheduler
    >>> )
    >>> trainer.run(n_epoch=10, loader=train_loader)
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        class MultiGPUTrainer(base_cls):
            def __init__(self, *args, init_method="tcp://localhost:12355", **kwargs):
                self.init_method = init_method
                super().__init__(*args, **kwargs)

            def run(
                self, n_epoch, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **aux_run_kwargs
            ):
                device_list = self.device if isinstance(self.device, list) else [self.device]
                mp.spawn(
                    self.train_process,
                    args=(device_list, self.init_method, n_epoch, loader, valid_loader, aux_run_kwargs),
                    nprocs=len(device_list),
                    join=True,
                )

            def train_process(self, rank, device_list, init_method, n_epoch, loader, valid_loader, aux_run_kwargs):
                # 프로세스 그룹 초기화
                dist.init_process_group(backend="nccl", init_method=init_method, world_size=len(device_list), rank=rank)
                torch.cuda.set_device(device_list[rank])

                # primary_device_index 설정
                self.primary_device_index = rank

                # 모델을 DDP로 래핑
                self.model = DDP(self.model.to(device_list[rank]), device_ids=[device_list[rank]])

                # 새로운 DataLoader를 생성하고 DistributedSampler를 설정
                train_sampler = td.DistributedSampler(loader.dataset, num_replicas=len(device_list), rank=rank)
                train_loader = td.DataLoader(
                    loader.dataset,
                    batch_size=loader.batch_size,
                    shuffle=False,
                    num_workers=loader.num_workers,
                    pin_memory=loader.pin_memory,
                    sampler=train_sampler,
                    collate_fn=loader.collate_fn,
                )

                if valid_loader:
                    valid_sampler = td.DistributedSampler(
                        valid_loader.dataset, num_replicas=len(device_list), rank=rank
                    )
                    valid_loader = td.DataLoader(
                        valid_loader.dataset,
                        batch_size=valid_loader.batch_size,
                        shuffle=False,
                        num_workers=valid_loader.num_workers,
                        pin_memory=valid_loader.pin_memory,
                        sampler=valid_sampler,
                        collate_fn=valid_loader.collate_fn,
                    )
                else:
                    valid_loader = None

                # 기존 run 메소드 호출
                super(MultiGPUTrainer, self).run(n_epoch, train_loader, valid_loader, **aux_run_kwargs)

                # 프로세스 그룹 해제
                dist.destroy_process_group()

        return MultiGPUTrainer
