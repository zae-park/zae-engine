import os
from typing import Type, Optional, List
import copy
import torch
import torch.multiprocessing as mp
import torch.utils.data as td
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from .core import AddOnBase
from .._trainer import T

def train_process(rank, n_epoch, device_list, model_class, model_state_dict, trainer_args, trainer_kwargs, loader, valid_loader, **kwargs):
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
    trainer.run(n_epoch, loader, valid_loader, **kwargs)

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
                self.args, self.kwargs = args, kwargs
                self.device_list = kwargs.pop("device", ["cuda:0"])
                self.num_devices = len(self.device_list)
                self.model = kwargs.pop("model", None)
                
                # Save the class and arguments to create Trainer instances later
                self.trainer_class = base_cls
                self.trainer_args = args
                self.trainer_kwargs = kwargs
                self.model_state_dict = self.model.state_dict()

            def _create_trainer_for_device(self, device):
                """
                Create a trainer instance for a specific device.
                """
                # Deep copy the model and load the state dict
                model_copy = copy.deepcopy(self.model)
                model_copy.load_state_dict(self.model_state_dict)
                model_copy = model_copy.to(device)

                # Create and return Trainer instance
                trainer = self.trainer_class(model=model_copy, device=device, *self.trainer_args, **self.trainer_kwargs)
                return trainer

            def run(self, n_epoch, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **kwargs):
                """
                Run training/testing process on all GPUs.
                """
                # Prepare arguments for train_process
                train_kwargs = {
                    'device_list': self.device_list,
                    'model_class': self.model.__class__,
                    'model_state_dict': self.model_state_dict,
                    'trainer_args': self.trainer_args,
                    'trainer_kwargs': self.trainer_kwargs,
                    'loader': loader,
                    'valid_loader': valid_loader,
                    **kwargs
                }

                # Use multiprocessing to run training on each device
                mp.spawn(
                    train_process,
                    args=(n_epoch, self.device_list, self.model.__class__, self.model_state_dict, self.trainer_args, self.trainer_kwargs, loader, valid_loader, kwargs),
                    nprocs=self.num_devices,
                    join=True
                )

        return MultiGPUTrainer
