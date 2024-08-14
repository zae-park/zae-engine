import argparse
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons.mpu import MultiGPUAddon

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

class DummyTrainer(Trainer):
    def __init__(self, model, device, optimizer=None, scheduler=None, mode="train", *args, **kwargs):
        super(DummyTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        # Here, we assume the device handling is already taken care of by Trainer class
        x, y = batch
        outputs = self.model(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        return {"loss": loss}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        # Same assumption for device handling
        x, y = batch
        outputs = self.model(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        return {"loss": loss}


if __name__ == "__main__":

    world_size = 2

    model = SimpleModel()
    optimizer_ = SGD(model.parameters(), lr=0.01)
    scheduler_ = StepLR(optimizer_, step_size=1, gamma=0.7)

    MultiGPUTrainer = DummyTrainer.add_on(MultiGPUAddon)
    
    trainer = MultiGPUTrainer(
        model=model,
        device=[torch.device(f"cuda:{i}") for i in range(world_size)],
        mode="train",
        optimizer=optimizer_,
        scheduler=scheduler_,
        init_method="tcp://10.63.0.3:12355"
    )

    dataset = SimpleDataset(10)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    trainer.run(n_epoch=3, loader=train_loader, valid_loader=valid_loader)

