import os
from typing import Union, Dict
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import MultiGPUAddon


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class CustomTrainer(Trainer):
    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x, y = batch
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        return {"loss": loss}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x, y = batch
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        return {"loss": loss}


def train_process(rank, device_list, init_method, model_class, model_state_dict, trainer_class, trainer_args, trainer_kwargs, n_epoch, loader, valid_loader, aux_run_kwargs):
    print(f'Initializing process {rank}...')

    os.environ['WORLD_SIZE'] = str(len(device_list))
    os.environ['RANK'] = str(rank)

    print(f'Process {rank} initializing process group...')
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=len(device_list), rank=rank)
    torch.cuda.set_device(device_list[rank])

    print(f'Process {rank} loading model...')
    model = model_class()
    model.load_state_dict(model_state_dict)
    model = model.to(device_list[rank])

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    trainer = trainer_class(model=model, device=device_list[rank], *trainer_args, **trainer_kwargs)

    print(f'Process {rank} running training...')
    trainer.run(n_epoch, loader, valid_loader, **aux_run_kwargs)

    # Add a check to confirm DDP is set up correctly
    assert isinstance(model, nn.parallel.DistributedDataParallel), (
        f"Process {rank}: Model is not wrapped with DistributedDataParallel"
    )

    dist.destroy_process_group()


class TestMultiGPUAddon(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.devices = (
            [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        # Generate some random data
        self.data = torch.randn(100, 10)
        self.labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(self.data, self.labels)
        self.loader = DataLoader(dataset, batch_size=10, shuffle=True)

    def test_multi_gpu_training(self):
        if not torch.cuda.is_available() or len(self.devices) < 2:
            self.skipTest("Multi-GPU test requires at least 2 GPUs.")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        trainer_w_mpu = CustomTrainer.add_on(MultiGPUAddon)
        trainer = trainer_w_mpu(
            model=self.model,
            device=self.devices,  # Initialize with multiple devices
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            init_method="tcp://localhost:12355"
        )

        trainer.run(n_epoch=1, loader=self.loader)

        print("Multi-GPU training test passed")


if __name__ == "__main__":
    unittest.main()
