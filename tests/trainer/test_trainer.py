import unittest
from random import randint
from typing import Union, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim

from zae_engine.models.foundations.resnet import resnet18
from zae_engine.trainer import Trainer
from zae_engine.schedulers import WarmUpScheduler


class ExDataset(Dataset):
    def __init__(self, n_data: int):
        super(ExDataset, self).__init__()
        self.n_data = n_data

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return idx


class ExTrainer(Trainer):
    def __init__(self, model, device, mode, scheduler, optimizer, *args, **kwargs):
        super(ExTrainer, self).__init__(model, device, mode, scheduler=scheduler, optimizer=optimizer, *args, **kwargs)

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return {"loss": torch.zeros(1)}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return self.train_step(batch)


class TestTrainer(unittest.TestCase):
    model = None
    optimizer = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = resnet18()
        cls.cpu = torch.device("cpu")
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.optimizer = optim.Adam(cls.model.parameters())
        cls.scheduler = WarmUpScheduler(cls.optimizer, total_iters=randint(0, 256))
        cls.dataset = Dataset()

    def setUp(self) -> None:
        self.dataset = ExDataset(n_data=randint(0, 256))
        self.trainer = ExTrainer(self.model, self.device, "train", scheduler=self.scheduler, optimizer=self.optimizer)

    def test_device_transfer(self):
        sample = torch.zeros(1, 3, 28, 28)
        dummy_sample = 1

        if torch.cuda.is_available():
            in_a_device = self.trainer._to_device(sample)
            in_are_device = self.trainer._to_device(sample, dummy_sample)
            self.assertEqual(in_a_device.get_device(), 0)
            for in_device in in_are_device:
                self.assertEqual(in_device.get_device(), 0)
        else:
            in_a_cpu = self.trainer._to_cpu(sample)
            self.assertEqual(in_a_cpu.get_device(), -1)

            in_cpu, dummy_in_cpu = self.trainer._to_cpu(sample, dummy_sample)
            self.assertEqual(in_cpu.get_device(), -1)
            self.assertEqual(id(dummy_sample), id(dummy_in_cpu))

    # def test_data_count(self):
    #       Test in Runner
    #     loader = DataLoader(self.dataset)
    #     self.trainer.loader = loader
    #
    #     pass

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
