import os
from typing import Union, Dict
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import MultiGPUAddon


class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.mean(dim=1) > 0.5).long()  # Label is 1 if mean > 0.5, otherwise 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define MultiGPUTrainer
class MultiGPUTrainer(Trainer.add_on(MultiGPUAddon)):
    def train_step(self, batch):
        data, labels = batch
        outputs = self.model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return {"loss": loss}

    def test_step(self, batch):
        return self.train_step(batch)


class TestMultiGPUAddon(unittest.TestCase):
    def setUp(self):
        self.device_list = (
            [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        )

        # Dataset and DataLoader
        dataset = DummyDataset(1000)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model = nn.Linear(10, 2)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def test_multi_gpu_training(self):
        if not torch.cuda.is_available() or len(self.devices) < 2:
            self.skipTest("Multi-GPU test requires at least 2 GPUs.")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        trainer = MultiGPUTrainer(
            model=self.model,
            device=self.device_list,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            init_method="tcp://localhost:12355",
        )

        trainer.run(n_epoch=2, loader=self.train_loader)

        print("Multi-GPU training test passed")


if __name__ == "__main__":
    unittest.main()
