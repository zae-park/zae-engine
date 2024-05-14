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
from zae_engine.models import utility


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
        return {"loss": torch.ones(1)}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        return self.train_step(batch)


class TestTrainer(unittest.TestCase):
    model = None
    optimizer = None
    scheduler = None
    n_data = None
    loader = None
    trainer = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = resnet18()
        cls.cpu = torch.device("cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.optimizer = optim.Adam(cls.model.parameters())
        cls.scheduler = WarmUpScheduler(cls.optimizer, total_iters=randint(0, 256))
        cls.n_data = randint(0, 256)
        dataset = ExDataset(n_data=cls.n_data)
        cls.trainer = ExTrainer(cls.model, device, "train", scheduler=cls.scheduler, optimizer=cls.optimizer)
        bs = randint(0, 256)
        cls.loader = DataLoader(dataset, batch_size=bs)
        cls.trainer.run(n_epoch=randint(0, 4), loader=cls.loader)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove("./test.pth")

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

    def test_run(self):
        # test _check_batch_size
        size = min(self.trainer.loader.batch_size, self.n_data)
        self.assertEqual(self.trainer.batch_size, size)
        self.assertEqual(self.trainer.valid_batch_size, 0)
        # test _scheduler_step_check
        pass
        # test logging
        self.assertEqual(self.trainer.batch_cnt, len(self.trainer.log_train))
        self.assertEqual(0, len(self.trainer.log_test))

    def test_steps(self):
        dummy_sample = [randint(0, 128)] * randint(0, 128)
        dummy_train = self.trainer.train_step({"dummy": dummy_sample})
        dummy_test = self.trainer.test_step({"dummy": dummy_sample})
        self.assertDictEqual(dummy_train, dummy_test)

    def test_toggle(self):
        toggle_count = randint(1, 256)
        for i in range(toggle_count):
            self.trainer.toggle()
        self.assertEqual(self.trainer.mode, "test" if toggle_count % 2 else "train")

        self.trainer.toggle("zae-park")
        self.assertEqual(self.trainer.mode, "zae-park")
        self.trainer.toggle()
        self.assertEqual(self.trainer.mode, "train")

    def test_check_better(self):
        pre_buffer = self.trainer.weight_buffer
        self.trainer.run(n_epoch=1, loader=self.loader)
        mid_buffer = self.trainer.weight_buffer
        self.trainer.train_step = lambda batch: {"loss": torch.zeros(1)}
        self.trainer.run(n_epoch=1, loader=self.loader)
        post_buffer = self.trainer.weight_buffer
        self.assertDictEqual(pre_buffer, mid_buffer)
        with self.assertRaises(AssertionError):
            self.assertDictEqual(mid_buffer, post_buffer)

    def test_log_reset(self):
        self.trainer.log_reset()
        self.assertFalse(self.trainer.log_train)
        self.assertFalse(self.trainer.log_test)

    def test_print_log(self):
        c_batch, n_batch = randint(0, 16), randint(0, 16)
        log_str, log_dict = self.trainer.print_log(cur_batch=c_batch, num_batch=n_batch)
        if log_dict["end"] and log_dict["file"] is None:
            self.assertEqual(c_batch, n_batch)
        else:
            self.assertNotEqual(c_batch, n_batch)
        self.assertIn(f"{c_batch}/{n_batch}", log_str)

    def test_save_model(self):
        self.trainer.save_model("./test.pth")
        dir_list = os.listdir(".")
        self.assertIn("test.pth", dir_list)

    def test_apply_weight(self):
        pre_weight = self.trainer.model.state_dict()
        self.trainer.model.apply(utility.initializer)
        mid_weight = self.trainer.model.state_dict()
        self.trainer.apply_weights("./test.pth", strict=True)
        post_weight = self.trainer.model.state_dict()
        self.assertDictEqual(pre_weight, post_weight)
        with self.assertRaises(AssertionError):
            self.assertDictEqual(mid_weight, post_weight)

    def test_inference(self):
        self.trainer.inference(loader=self.loader)
        self.assertEqual(self.trainer.valid_batch_size, len(self.trainer.log_test))
        self.assertEqual(0, len(self.trainer.log_train))


if __name__ == "__main__":
    unittest.main()
