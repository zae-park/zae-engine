import unittest
import os
import math
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
        return {"loss": torch.ones(1, requires_grad=True), "output": torch.zeros(1, requires_grad=True)}

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
        bs = randint(1, 256)
        cls.loader = DataLoader(dataset, batch_size=bs)

    @classmethod
    def tearDownClass(cls) -> None:
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
        # test _check_batch_size
        self.assertEqual(self.trainer.batch_size, self.trainer.loader.batch_size)
        self.assertEqual(self.trainer.valid_batch_size, 0)

        # test logging
        self.assertTrue(self.trainer.train_metrics)  # Check that metrics were computed
        self.assertEqual(len(self.loader), len(self.trainer.log_train["loss"]))
        self.assertEqual(0, len(self.trainer.log_test["loss"]))

        # test batch step
        self.trainer.scheduler_step_on_batch = True
        test_epoch = randint(0, 4)
        if test_epoch * len(self.loader) > self.trainer.scheduler.total_iters:
            with self.assertRaises(AssertionError):
                self.trainer.run(n_epoch=test_epoch, loader=self.loader)
        else:
            self.trainer.run(n_epoch=test_epoch, loader=self.loader)

    def test_single_epoch_run(self):
        self.trainer.metric_on_epoch_end = lambda: {"asd": 1}
        self.trainer.run_epoch(loader=self.loader)
        # test metrics
        print(f"Train metrics: {self.trainer.train_metrics}")
        print(f"Valid metrics: {self.trainer.test_metrics}")
        self.assertTrue(self.trainer.train_metrics)  # Check train metrics exist
        self.assertFalse(self.trainer.test_metrics)  # Valid metrics should be empty in this case

        self.trainer.run_epoch(loader=self.loader)
        self.trainer.mode = "test"
        self.trainer.run_epoch(loader=self.loader)
        # test metrics
        print(f"Train metrics: {self.trainer.train_metrics}")
        print(f"Valid metrics: {self.trainer.test_metrics}")
        self.assertTrue(self.trainer.train_metrics)  # Check train metrics exist
        self.assertTrue(self.trainer.test_metrics)  # Valid metrics should be empty in this case

    def test_steps(self):
        dummy_sample = [randint(0, 128)] * randint(0, 128)
        dummy_train = self.trainer.train_step({"dummy": dummy_sample})
        dummy_test = self.trainer.test_step({"dummy": dummy_sample})
        self.assertDictEqual(dummy_train, dummy_test)

    def test_toggle(self):
        pre_mode = self.trainer.mode
        toggle_count = randint(1, 256)
        for i in range(toggle_count):
            self.trainer.toggle()
        if toggle_count % 2:
            self.assertNotEqual(self.trainer.mode, pre_mode)
        else:
            self.assertEqual(self.trainer.mode, pre_mode)

        self.trainer.toggle("zae-park")
        self.assertEqual(self.trainer.mode, "zae-park")
        self.trainer.toggle()
        self.assertEqual(self.trainer.mode, "train")

    def test_check_better(self):
        pre_buffer = max(1, len(self.trainer.weight_buffer["epoch"]))
        self.trainer.run(n_epoch=1, loader=self.loader)
        mid_buffer = len(self.trainer.weight_buffer["epoch"])
        self.trainer.train_step = lambda batch: {"loss": torch.zeros(1, requires_grad=True)}
        self.trainer.run(n_epoch=1, loader=self.loader)
        post_buffer = len(self.trainer.weight_buffer["epoch"])
        self.assertEqual(pre_buffer, mid_buffer)
        self.assertGreater(post_buffer, pre_buffer)

    def test_log_reset(self):
        self.trainer.log_reset()
        self.assertFalse(self.trainer.log_train)
        self.assertFalse(self.trainer.log_test)
        self.assertFalse(self.trainer.train_metrics)  # Ensure train metrics are reset
        self.assertFalse(self.trainer.test_metrics)  # Ensure valid metrics are reset

    def test_print_log(self):
        c_batch, n_batch = randint(0, 16), randint(0, 16)
        log_str, log_dict = self.trainer.print_log(cur_batch=c_batch, num_batch=n_batch)
        if log_dict["end"] and log_dict["file"] is None:
            self.assertEqual(c_batch, n_batch)
        else:
            self.assertNotEqual(c_batch, n_batch)
        self.assertIn(f"{c_batch}/{n_batch}", log_str)

    def test_model_save_load(self):
        dummy_sample = self.trainer._to_device(torch.ones((1, 3, 256, 256)), dtype=torch.float32)

        # test save_model
        self.trainer.save_model("./test.pth")
        dir_list = os.listdir(".")
        self.assertIn("test.pth", dir_list)

        # test apply_weight
        pre_result = self.trainer.model(dummy_sample)
        self.trainer.model.apply(utility.initializer)
        mid_result = self.trainer.model(dummy_sample)
        self.trainer.apply_weights("./test.pth", strict=True)
        post_result = self.trainer.model(dummy_sample)
        self.assertEqual(self.trainer._to_cpu(pre_result).sum(), self.trainer._to_cpu(post_result).sum())
        self.assertNotEqual(self.trainer._to_cpu(mid_result).sum(), self.trainer._to_cpu(post_result).sum())

    def test_inference(self):
        # Perform training to populate metrics (log_train/log_test initialization irrelevant)
        self.trainer.run(n_epoch=1, loader=self.loader)
        pre_train_metrics = self.trainer.train_metrics.copy()
        pre_test_metrics = self.trainer.test_metrics.copy()
        original_mode = self.trainer.mode

        # Run inference
        inference_results = self.trainer.inference(loader=self.loader)

        # Verify inference results do not modify training metrics
        self.assertEqual(self.trainer.train_metrics, pre_train_metrics)
        self.assertEqual(self.trainer.test_metrics, pre_test_metrics)

        # Verify inference results are consistent
        self.assertGreater(len(inference_results), 0)  # Ensure inference produced results
        batch_cnt = math.ceil(self.n_data / self.loader.batch_size)
        self.assertEqual(len(self.trainer.log_test.get("output", [])), batch_cnt)

        # Verify mode is restored after inference
        self.assertEqual(self.trainer.mode, original_mode)


if __name__ == "__main__":
    unittest.main()
