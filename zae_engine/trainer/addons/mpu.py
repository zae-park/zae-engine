import os
import copy
from typing import Type, Optional, List
import torch
import torch.multiprocessing as mp
import torch.utils.data as td
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from .core import AddOnBase
from .._trainer import T

def train_process(rank, device_list, model_class, model_state_dict, trainer_class, trainer_args, trainer_kwargs, n_epoch, loader, valid_loader, aux_run_kwargs):
    print('1:', rank, device_list)
    print('2:', model_class, model_state_dict)
    print('3:', trainer_class, trainer_args, trainer_kwargs)
    print('4:', n_epoch, loader, valid_loader, aux_run_kwargs)
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(device_list), rank=rank)
    torch.cuda.set_device(device_list[rank])

    # Create a new model instance and load state dict
    model = model_class()
    model.load_state_dict(model_state_dict)
    model = model.to(device_list[rank])
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create Trainer instance for the current process
    trainer = trainer_class(model=model, device=device_list[rank], *trainer_args, **trainer_kwargs)
    
    # Assign DistributedSampler to loaders
    loader.sampler = DistributedSampler(loader.dataset, num_replicas=len(device_list), rank=rank)
    if valid_loader:
        valid_loader.sampler = DistributedSampler(valid_loader.dataset, num_replicas=len(device_list), rank=rank)
    
    # Run training
    trainer.run(n_epoch, loader, valid_loader, **aux_run_kwargs)

class MultiGPUAddon(AddOnBase):
    """
    Add-on for enabling multi-GPU support by distributing the model across multiple GPUs
    and creating separate Trainer instances for each GPU.
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        """
        Applies the multi-GPU modifications to the base class.
        """
        class MultiGPUTrainer:
            def __init__(self, *args, **kwargs):
                self.device_list = kwargs.pop("device", ["cuda:0"])
                self.num_devices = len(self.device_list)
                self.model = kwargs.pop("model", None)
                
                # Save the class and arguments to create Trainer instances later
                self.trainer_class = base_cls
                self.trainer_args = args
                self.trainer_kwargs = kwargs
                self.model_state_dict = self.model.state_dict()

            def run(self, n_epoch, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **aux_run_kwargs):
                """
                Run training/testing process on all GPUs.
                """
                # Prepare arguments for train_process
                model_args = (self.model.__class__, self.model_state_dict)
                trainer_args = (self.trainer_class, self.trainer_args, self.trainer_kwargs)
                run_args = (n_epoch, loader, valid_loader, aux_run_kwargs)
                args = (self.device_list, ) + model_args + trainer_args + run_args

                # Use multiprocessing to run training on each device
                mp.spawn(
                    train_process,
                    args=args,
                    nprocs=self.num_devices,
                    join=True
                )

        return MultiGPUTrainer
