import unittest
import os
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
        return {"loss": torch.ones(1, requires_grad=True), "output": torch.zeros(1, requires_grad=True)}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return self.train_step(batch)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet18()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = WarmUpScheduler(self.optimizer, total_iters=randint(0, 256))
        dataset = ExDataset(n_data=randint(0, 256))
        self.loader = DataLoader(dataset, batch_size=randint(1, 256))
        self.trainer = ExTrainer(self.model, device, "train", scheduler=self.scheduler, optimizer=self.optimizer)
        self.trainer.log_reset(epoch_end=True)

    def tearDown(self):
        if os.path.exists("./test.pth"):
            os.remove("./test.pth")

    def test_device_transfer(self):
        sample = torch.zeros(1, 3, 28, 28)
        dummy_sample = 1

        if torch.cuda.is_available():
            in_a_device = self.trainer._to_device(sample)
            in_device, dummy_in_device = self.trainer._to_cpu(sample, dummy_sample)
            self.assertEqual(in_a_device.get_device(), 0)
            self.assertEqual(in_device.get_device(), -1)
            self.assertEqual(id(dummy_sample), id(dummy_in_device))
        else:
            in_a_cpu = self.trainer._to_cpu(sample)
            self.assertEqual(in_a_cpu.get_device(), -1)

            in_cpu, dummy_in_cpu = self.trainer._to_cpu(sample, dummy_sample)
            self.assertEqual(in_cpu.get_device(), -1)
            self.assertEqual(id(dummy_sample), id(dummy_in_cpu))

    def test_run(self):
        self.trainer.toggle("train")
        self.trainer.metric_on_epoch_end = lambda: {"asd": 1}
        self.trainer.run(n_epoch=randint(1, 4), loader=self.loader)
        self.assertEqual(self.trainer.batch_size, self.loader.batch_size)
        self.assertEqual(self.trainer.valid_batch_size, 0)
        self.assertTrue(self.trainer.train_metrics)
        self.assertEqual(len(self.loader), len(self.trainer.log_train["loss"]))

    def test_inference(self):
        self.trainer.run(n_epoch=1, loader=self.loader)
        pre_train_metrics = self.trainer.train_metrics.copy()
        pre_test_metrics = self.trainer.test_metrics.copy()
        original_mode = self.trainer.mode

        inference_results = self.trainer.inference(loader=self.loader)

        self.assertEqual(self.trainer.train_metrics, pre_train_metrics)
        self.assertEqual(self.trainer.test_metrics, pre_test_metrics)
        self.assertGreater(len(inference_results), 0)
        batch_cnt = len(self.loader)
        self.assertEqual(len(self.trainer.log_test.get("output", [])), batch_cnt)
        self.assertEqual(self.trainer.mode, original_mode)

    def test_log_reset(self):
        self.trainer.log_reset()
        self.assertFalse(self.trainer.log_train)
        self.assertFalse(self.trainer.log_test)
        self.assertFalse(self.trainer.train_metrics)
        self.assertFalse(self.trainer.test_metrics)


if __name__ == "__main__":
    unittest.main()
