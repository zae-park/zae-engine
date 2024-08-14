import os
import tempfile
from typing import Type, Union, Optional, Dict

import torch
import torch.utils.data as td
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .core import AddOnBase
from .._trainer import T


def train_process(rank, device_list, model_class, model_state_dict, trainer_class, trainer_args, trainer_kwargs, n_epoch, loader, valid_loader, aux_run_kwargs):
    print(f'Initializing process {rank}...')
    
    # Create a temporary file to use for the file-based rendezvous
    init_file = tempfile.NamedTemporaryFile(delete=True)
    init_file.close()  # We close the file so that other processes can access it
    
    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(len(device_list))
    os.environ['RANK'] = str(rank)
    
    print(f'Process {rank} initializing process group...')
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method=f'file://{init_file.name}', world_size=len(device_list), rank=rank)
    torch.cuda.set_device(device_list[rank])

    # Create a new model instance and load state dict
    print(f'Process {rank} loading model...')
    model = model_class()
    model.load_state_dict(model_state_dict)
    model = model.to(device_list[rank])
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create Trainer instance for the current process
    trainer = trainer_class(model=model, device=device_list[rank], *trainer_args, **trainer_kwargs)
    
    # Assign DistributedSampler to loaders
    print(f'Process {rank} setting up data loaders...')
    loader_sampler = DistributedSampler(loader.dataset, num_replicas=len(device_list), rank=rank)
    train_loader_mp = td.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,  # shuffle=False since DistributedSampler handles shuffling
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        sampler=loader_sampler
    )
    
    valid_loader_mp = None
    if valid_loader:
        valid_loader_sampler = DistributedSampler(valid_loader.dataset, num_replicas=len(device_list), rank=rank)
        valid_loader_mp = td.DataLoader(
            valid_loader.dataset,
            batch_size=valid_loader.batch_size,
            shuffle=False,  # shuffle=False since DistributedSampler handles shuffling
            num_workers=valid_loader.num_workers,
            pin_memory=valid_loader.pin_memory,
            drop_last=valid_loader.drop_last,
            sampler=valid_loader_sampler
        )
    
    # Run training
    print(f'Process {rank} running training...')
    trainer.run(n_epoch, train_loader_mp, valid_loader_mp, **aux_run_kwargs)


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
