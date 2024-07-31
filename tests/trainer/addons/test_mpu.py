import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import MultiGPUAddon


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestMultiGPUAddon(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.devices = (
            [torch.device(f"cuda:0"), torch.device(f"cuda:{torch.cuda.device_count() - 1}")]
            if torch.cuda.is_available()
            else []
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        # Generate some random data
        self.data = torch.randn(100, 10)
        self.labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(self.data, self.labels)
        self.loader = DataLoader(dataset, batch_size=10, shuffle=True)

    def test_multi_gpu_training(self):
        TrainerWithMultiGPU = Trainer.add_on(MultiGPUAddon)
        trainer = TrainerWithMultiGPU(
            model=self.model,
            device=self.devices,  # Initialize with multiple devices
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        trainer.run(n_epoch=1, loader=self.loader)
        self.assertTrue(isinstance(trainer.model, nn.DataParallel), "Model is not wrapped with DataParallel")
        print("Multi-GPU training test passed")


if __name__ == "__main__":
    unittest.main()
