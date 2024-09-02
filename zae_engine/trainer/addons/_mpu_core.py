# multi_gpu_trainer.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Type, Optional, Union, Dict

from .core import AddOnBase
from .._trainer import Trainer
from .._trainer import T


class MultiGPUTrainer(Trainer):
    def __init__(self, *args, init_method="tcp://localhost:12355", device_list=None, **kwargs):
        self.init_method = init_method
        self.device_list = device_list or ["cuda:0"]
        super().__init__(*args, **kwargs)

    def run(
        self,
        n_epoch,
        loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
        **aux_run_kwargs,
    ):
        mp.spawn(
            self.train_process,
            args=(self.device_list, self.init_method, n_epoch, loader, valid_loader, aux_run_kwargs),
            nprocs=len(self.device_list),
            join=True,
        )

    def train_process(self, rank, device_list, init_method, n_epoch, loader, valid_loader, aux_run_kwargs):
        # 프로세스 그룹 초기화
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=len(device_list), rank=rank)
        torch.cuda.set_device(device_list[rank])

        # 모델을 DDP로 래핑
        self.model = DDP(self.model.to(device_list[rank]), device_ids=[device_list[rank]])

        # DistributedSampler와 DataLoader 생성
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            loader.dataset, num_replicas=len(device_list), rank=rank
        )
        train_loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            sampler=train_sampler,
        )

        if valid_loader:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_loader.dataset, num_replicas=len(device_list), rank=rank
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_loader.dataset,
                batch_size=valid_loader.batch_size,
                shuffle=False,
                num_workers=valid_loader.num_workers,
                pin_memory=valid_loader.pin_memory,
                sampler=valid_sampler,
            )
        else:
            valid_loader = None

        # 학습 수행
        super(MultiGPUTrainer, self).run(n_epoch, train_loader, valid_loader, **aux_run_kwargs)

        # 프로세스 그룹 해제
        dist.destroy_process_group()
