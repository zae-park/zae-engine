import unittest
import os
import shutil
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import StateManagerAddon


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


class TestStateManagerAddon(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.save_path = "test_saves"

        # Generate some random data
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        self.loader = DataLoader(dataset, batch_size=10, shuffle=True)

    def tearDown(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

    def test_state_saving(self):
        trainer_cls = CustomTrainer.add_on(StateManagerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=self.device,
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_path=self.save_path,
            save_format="ckpt",
        )

        trainer.run(n_epoch=1, loader=self.loader)
        trainer.save_state()
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "model.ckpt")), "Model checkpoint not saved.")
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "optimizer.zae")), "Optimizer state not saved.")
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "scheduler.zae")), "Scheduler state not saved.")

    def test_state_loading(self):
        trainer_cls = CustomTrainer.add_on(StateManagerAddon)
        trainer = trainer_cls(
            model=self.model,
            device=self.device,
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_path=self.save_path,
            save_format="ckpt",
        )

        trainer.run(n_epoch=1, loader=self.loader)
        trainer.save_state()
        trainer.load_state()

        self.assertTrue(os.path.exists(os.path.join(self.save_path, "model.ckpt")), "Model checkpoint not saved.")
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "optimizer.zae")), "Optimizer state not saved.")
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "scheduler.zae")), "Scheduler state not saved.")


if __name__ == "__main__":
    unittest.main()
