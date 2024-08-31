import os
from typing import Type, Optional
import torch
import torch.utils.data as td
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .core import AddOnBase
from .._trainer import T


def default_train_process(rank, world_size, model_class, model_state_dict, trainer_class, trainer_args, trainer_kwargs):
    """
    Default training process that uses the Trainer's run method.
    
    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        Total number of GPUs being used.
    model_class : Type[torch.nn.Module]
        The class of the model to be trained.
    model_state_dict : dict
        The state dictionary of the model to load.
    trainer_class : Type[Trainer]
        The class of the trainer to be used.
    trainer_args : tuple
        The positional arguments for the trainer.
    trainer_kwargs : dict
        The keyword arguments for the trainer.
    """
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:12355', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    
    # Initialize model
    model = model_class()
    model.load_state_dict(model_state_dict)
    model = model.to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Initialize trainer
    trainer = trainer_class(model=model, device=torch.device(f'cuda:{rank}'), *trainer_args, **trainer_kwargs)
    
    # Run training
    trainer.run(n_epoch=2, loader=trainer.loader)
    
    dist.destroy_process_group()

class MultiGPUAddon(AddOnBase):
    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        class MultiGPUTrainer(base_cls):
            def __init__(self, *args, init_method="tcp://localhost:12355", **kwargs):
                self.init_method = init_method
                super().__init__(*args, **kwargs)

            def run(self, n_epoch, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **aux_run_kwargs):
                device_list = self.device if isinstance(self.device, list) else [self.device]
                mp.spawn(
                    self.train_process,
                    args=(device_list, self.init_method, n_epoch, loader, valid_loader, aux_run_kwargs),
                    nprocs=len(device_list),
                    join=True
                )

            def train_process(self, rank, device_list, init_method, n_epoch, loader, valid_loader, aux_run_kwargs):
                # 프로세스 그룹 초기화
                dist.init_process_group(backend='nccl', init_method=init_method, world_size=len(device_list), rank=rank)
                torch.cuda.set_device(device_list[rank])

                # 모델을 DDP로 래핑
                self.model = DDP(self.model.to(device_list[rank]), device_ids=[device_list[rank]])

                # 데이터 로더의 분산 샘플러 설정
                self.loader = loader
                self.valid_loader = valid_loader

                # Trainer의 기존 run 메소드 호출
                super(MultiGPUTrainer, self).run(n_epoch, loader, valid_loader, **aux_run_kwargs)

        return MultiGPUTrainer


