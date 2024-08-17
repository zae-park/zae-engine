import os
from typing import Type, Optional, Dict

import torch
import torch.utils.data as td
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .core import AddOnBase
from .._trainer import T


def train_process(rank, device_list, init_method, model_class, model_state_dict, trainer_class, trainer_args, trainer_kwargs, n_epoch, loader, valid_loader, aux_run_kwargs):
    """
    Initializes and runs a distributed training process for a single GPU.

    This function is executed in a separate process for each GPU. It sets up the
    distributed training environment, initializes the model, and performs training.

    Parameters
    ----------
    rank : int
        The rank of the current process, used to identify the GPU and setup distributed training.
    device_list : list of str
        List of GPU device strings (e.g., ["cuda:0", "cuda:1"]).
    init_method : str
        The initialization method for distributed training. This should be a TCP address
        and port in the format "tcp://<MASTER_IP>:<PORT>".
    model_class : Type[torch.nn.Module]
        The model class to be instantiated and trained. This class must be compatible with
        `torch.nn.Module`.
    model_state_dict : dict
        The state dictionary of the model to be loaded into the model instance.
    trainer_class : Type
        The Trainer class used for managing the training process.
    trainer_args : tuple
        Positional arguments to be passed to the Trainer class.
    trainer_kwargs : dict
        Keyword arguments to be passed to the Trainer class.
    n_epoch : int
        The number of epochs for training.
    loader : torch.utils.data.DataLoader
        The training data loader. This should provide the training data in batches.
    valid_loader : Optional[torch.utils.data.DataLoader]
        The validation data loader, if any. If not provided, validation is skipped.
    aux_run_kwargs : dict
        Additional keyword arguments to be passed to the `run` method of the Trainer class.

    Notes
    -----
    This function creates a temporary file to facilitate the initialization of the distributed
    process group. The file is used to synchronize the setup of the distributed environment
    across all processes. After setting up the distributed environment, the model is instantiated,
    loaded with the provided state dictionary, and wrapped with `DistributedDataParallel` (DDP).
    The training process is then run using the provided Trainer class and data loaders.

    Example
    -------
    >>> train_process(
    >>>     rank=0,
    >>>     device_list=["cuda:0", "cuda:1"],
    >>>     init_method="tcp://localhost:12355",
    >>>     model_class=MyModel,
    >>>     model_state_dict=my_model_state_dict,
    >>>     trainer_class=MyTrainer,
    >>>     trainer_args=(arg1, arg2),
    >>>     trainer_kwargs={'kwarg1': value1},
    >>>     n_epoch=10,
    >>>     loader=train_loader,
    >>>     valid_loader=valid_loader,
    >>>     aux_run_kwargs={}
    >>> )
    """
    
    print(f'Initializing process {rank}...')
    
    os.environ['WORLD_SIZE'] = str(len(device_list))
    os.environ['RANK'] = str(rank)
    
    print(f'Process {rank} initializing process group...')
    # Initialize distributed process group with TCP initialization method
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=len(device_list), rank=rank)
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

    This add-on is designed to facilitate distributed training across multiple GPUs by:
    - Setting up a distributed training environment using PyTorch's DistributedDataParallel (DDP).
    - Initializing a separate training process for each GPU.
    - Supporting distributed data loading with `DistributedSampler`.

    The `MultiGPUAddon` class works with a `Trainer` class that performs the actual training
    and evaluation. This add-on ensures that each GPU operates in a separate process and 
    uses a shared training setup.

    Example
    -------
    >>> class MyTrainer:
    >>>     # Define your trainer class
    >>>     pass
    >>>
    >>> multi_gpu_trainer = MultiGPUAddon.apply(MyTrainer)
    >>> trainer_instance = multi_gpu_trainer(
    >>>     device=["cuda:0", "cuda:1"],  # List of GPU devices to use
    >>>     model=my_model,  # Model instance
    >>>     init_method="tcp://<MASTER_IP>:<PORT>",  # Initialization method for distributed training
    >>>     # Any other arguments required by MyTrainer
    >>> )
    >>> trainer_instance.run(
    >>>     n_epoch=10,
    >>>     loader=train_loader,
    >>>     valid_loader=valid_loader,
    >>>     # Any additional arguments for the run method
    >>> )
    """

    @classmethod
    def apply(cls, base_cls: Type[T]) -> Type[T]:
        """
        Applies the multi-GPU modifications to the base class.

        This method wraps the given Trainer class with multi-GPU support, enabling distributed
        training across multiple GPUs. The returned class can be instantiated and used for
        training models across multiple GPUs.

        Parameters
        ----------
        base_cls : Type[T]
            The Trainer class to be enhanced with multi-GPU support.

        Returns
        -------
        Type[T]
            A new Trainer class with multi-GPU support.

        Raises
        ------
        NotImplementedError
            If the `init_method` provided is not supported.
        """
        class MultiGPUTrainer:
            """
            A trainer class for running distributed training across multiple GPUs.

            This class manages the setup and execution of training processes on each GPU.
            It initializes the distributed environment, sets up data loaders with distributed
            sampling, and runs the training process on all specified GPUs.

            Parameters
            ----------
            device : list of str
                List of GPU device strings (e.g., ["cuda:0", "cuda:1"]).
            model : torch.nn.Module
                The model to be trained.
            init_method : str
                The initialization method for distributed training (must start with "tcp").
            trainer_class : Type[T]
                The Trainer class to use for training.
            trainer_args : tuple
                Positional arguments for the Trainer class.
            trainer_kwargs : dict
                Keyword arguments for the Trainer class.

            Methods
            -------
            run(n_epoch, loader, valid_loader=None, **aux_run_kwargs)
                Runs the training/testing process on all GPUs.

                Parameters
                ----------
                n_epoch : int
                    The number of epochs to train for.
                loader : torch.utils.data.DataLoader
                    The training data loader.
                valid_loader : Optional[torch.utils.data.DataLoader]
                    The validation data loader, if any.
                **aux_run_kwargs
                    Additional arguments for the training run.
            """

            def __init__(self, *args, **kwargs):
                """
                Initializes the MultiGPUTrainer instance.

                Parameters
                ----------
                *args
                    Positional arguments for the Trainer class.
                **kwargs
                    Keyword arguments for the Trainer class, including:
                    - device (list of str): List of GPU devices to use (e.g., ["cuda:0", "cuda:1"]).
                    - model (torch.nn.Module): The model to be trained.
                    - init_method (str): Initialization method for distributed training.
                """
                self.device_list = kwargs.pop("device", ["cuda:0"])
                self.num_devices = len(self.device_list)
                self.model = kwargs.pop("model", None)
                self.init_method = kwargs.pop("init_method", "tcp://localhost:12355")  # Default to TCP method
                
                if not self.init_method.startswith("tcp"):
                    raise NotImplementedError("MultiGPUAddon supports TCP initialization method only.")
                
                # Save the class and arguments to create Trainer instances later
                self.trainer_class = base_cls
                self.trainer_args = args
                self.trainer_kwargs = kwargs
                self.model_state_dict = self.model.state_dict()

            def run(self, n_epoch, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **aux_run_kwargs):
                """
                Runs the training process on all GPUs.

                Initializes the distributed environment and spawns separate processes for each GPU.
                Each process handles training on its assigned GPU.

                Parameters
                ----------
                n_epoch : int
                    The number of epochs to train for.
                loader : torch.utils.data.DataLoader
                    The training data loader.
                valid_loader : Optional[torch.utils.data.DataLoader]
                    The validation data loader, if any.
                **aux_run_kwargs
                    Additional arguments for the training run.
                """
                # Prepare arguments for train_process
                model_args = (self.model.__class__, self.model_state_dict)
                trainer_args = (self.trainer_class, self.trainer_args, self.trainer_kwargs)
                run_args = (n_epoch, loader, valid_loader, aux_run_kwargs)
                args = (self.device_list, self.init_method, ) + model_args + trainer_args + run_args

                # Use multiprocessing to run training on each device
                mp.spawn(
                    train_process,
                    args=args,
                    nprocs=self.num_devices,
                    join=True
                )

        return MultiGPUTrainer
