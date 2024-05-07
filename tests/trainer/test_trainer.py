import unittest
from random import randint
from typing import Union, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from zae_engine.models.foundations.resnet import resnet18
from zae_engine.trainer import Trainer


class ExTrainer(Trainer):
    def __init__(self, model, device, mode, scheduler, optimizer, *args, **kwargs):
        super(ExTrainer, self).__init__(model, device, mode, scheduler=scheduler, optimizer=optimizer, *args, **kwargs)

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return {"loss": torch.zeros(1)}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return self.train_step(batch)


class TestTrainer(unittest.TestCase):
    ex_dataset = Dataset()

    ex_model = resnet18()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ex_scheduler = torch.optim.lr_scheduler
    ex_optimizer = torch.optim.Adam
    trainer = ExTrainer(ex_model, device, mode="train", scheduler=ex_scheduler, optimizer=ex_optimizer)

    def setUp(self) -> None:
        self.device_check = torch.cuda.is_available()

    def test_device_transfer(self):
        sample = torch.zeros(1, 3, 28, 28)
        in_device = self.trainer._to_device(sample)
        if self.device_check:
            self.assertEqual(in_device.get_device(), 0)
        in_cpu = self.trainer._to_cpu(in_device)
        self.assertEqual(in_device.get_device(), -1)

    def test_data_count(self):
        # ex_loader = torch.utils.data
        pass

    def test_toggle(self):
        toggle_count = randint(1, 256)
        for i in range(toggle_count):
            self.trainer.toggle()
        self.assertEqual(self.trainer.mode, "test" if toggle_count % 2 else "train")

        self.trainer.toggle("zae-park")
        self.assertEqual(self.trainer.mode, "zae-park")
        self.trainer.toggle()
        self.assertEqual(self.trainer.mode, "train")

    def test_log_reset(self):
        self.trainer.log_reset()
        self.assertFalse(self.trainer.log_train)
        self.assertFalse(self.trainer.log_test)

    def test_model_save(self):
        pass

    def test_apply_weight(self):
        pass


if __name__ == "__main__":
    unittest.main()
