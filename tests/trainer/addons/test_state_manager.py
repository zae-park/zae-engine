import os
import shutil
from typing import Union, Dict

import unittest
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
        self.checkpoint_dir = "./checkpoints"
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)

        # Generate some random data
        self.data = torch.randn(100, 10)
        self.labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(self.data, self.labels)
        self.loader = DataLoader(dataset, batch_size=10, shuffle=True)

    def test_state_saving_loading(self):
        trainer_with_saver = CustomTrainer.add_on(StateManagerAddon)
        trainer = trainer_with_saver(
            model=self.model,
            device=self.device,
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir=self.checkpoint_dir,
            save_model_format="ckpt",
        )

        trainer.run(n_epoch=1, loader=self.loader)
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "model_1.ckpt")))
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "optimizer.zae")))
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "scheduler.zae")))

        # Create new Trainer instance & load state
        new_trainer = trainer_with_saver(
            model=self.model,
            device=self.device,
            mode="train",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir=self.checkpoint_dir,
            save_model_format="ckpt",
        )
        new_trainer.load_state(epoch=1)

        # Load model, optimizer, scheduler for instance of Trainer and check progress
        new_trainer.run(n_epoch=1, loader=self.loader)
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "model_2.ckpt")))


if __name__ == "__main__":
    unittest.main()
