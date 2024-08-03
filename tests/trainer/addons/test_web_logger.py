import os
import shutil
import unittest
from typing import Union, Dict
from datetime import datetime

import wandb
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import neptune.new as neptune
from neptune.new.exceptions import NeptuneInvalidApiTokenException

from zae_engine.models import DummyModel
from zae_engine.trainer import Trainer
from zae_engine.trainer.addons import NeptuneLoggerAddon, WandBLoggerAddon


class DummySet(Dataset):
    def __init__(self, dummy):
        self.dummy = dummy

    def __len__(self):
        return self.dummy.shape[0]

    def __getitem__(self, idx):
        return self.dummy[idx].float()


class DummyTrainer(Trainer):
    def __init__(self, model, optimizer=None, scheduler=None, mode="train"):
        super(DummyTrainer, self).__init__(model, torch.device("cpu"), mode, optimizer, scheduler)

    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x = torch.concat(batch).unsqueeze(1)
        logit = self.model(x)
        return {"loss": logit.sum(), "acc": 0.99}

    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        x = torch.concat(batch).unsqueeze(1)
        logit = self.model(x)
        return {"loss": logit.sum(), "acc": 0.99}


class TestLogger(unittest.TestCase):
    model = None
    optimizer = None
    scheduler = None
    train_loader = None
    valid_loader = None

    @classmethod
    def setUpClass(cls) -> None:
        os.mkdir("../wandb")
        dummy = torch.randn(10, 1, 2560)
        train_set = DummySet(dummy)
        valid_set = DummySet(dummy)

        cls.model = DummyModel()
        cls.train_loader = DataLoader(
            train_set,
            batch_size=2,
            shuffle=True,
        )

        cls.valid_loader = DataLoader(valid_set, batch_size=2, shuffle=False)
        cls.optimizer = Adam(cls.model.parameters(), lr=1e-6)
        cls.scheduler = torch.optim.lr_scheduler.LambdaLR(cls.optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("../wandb")

    @classmethod
    def get_attribute(cls):
        model, optimizer, scheduler = cls.model, cls.optimizer, cls.scheduler
        train_loader, valid_loader = cls.train_loader, cls.valid_loader
        return model, optimizer, scheduler, train_loader, valid_loader

    def setUp(self) -> None:
        now = datetime.now()
        self.step_check = np.random.randint(1, 10)
        self.epoch_check = np.random.randint(1, 3)
        self.model, self.optimizer, self.scheduler, self.train_loader, self.valid_loader = self.get_attribute()
        self.test_time_point = str(now.timestamp()).split(".")[0][:9]

        # Initialize WandB
        wandb.setup(wandb.Settings(program="test_callback.py", program_relpath="test_callback.py"))
        try:
            self.wandb_runner = wandb.init(project="wandb-test", config={"a": 1, "b": 2, "c": 3})
        except wandb.errors.UsageError as e:
            self.skipTest(f"WandB initialization failed: {e}")

        # Initialize Neptune
        self.neptune_run = None
        try:
            self.neptune_run = neptune.init_run(project="test_project", api_token="your_neptune_api_token")
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(f"Neptune initialization failed: {e}")

    def tearDown(self) -> None:
        self.step_check = None
        self.epoch_check = None
        self.test_time_point = None

        if self.wandb_runner:
            self.wandb_runner.finish()
            self.wandb_runner = None

        if self.neptune_run:
            self.neptune_run.stop()
            self.neptune_run = None

    def test_wandb_init(self):
        self.assertIn("wandb", os.listdir(".."))

    def test_wandb_log(self):
        self.wandb_runner.log({"test": True})
        self.assertTrue(self.wandb_runner.summary["test"])

    def test_neptune_init(self):
        web_logger = {
            "neptune": {
                "project_name": "test_project",
                "api_tkn": "your_neptune_api_token",
            }
        }
        trainer_with_neptune = DummyTrainer.add_on(NeptuneLoggerAddon)
        trainer = trainer_with_neptune(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            neptune_config=web_logger["neptune"],
        )
        self.assertTrue(trainer.neptune_run is not None)

    def test_combined_logger(self):
        web_logger = {
            "wandb": {
                "project": "wandb-test",
                "config": {"a": 1, "b": 2, "c": 3},
            },
            "neptune": {
                "project_name": "test_project",
                "api_tkn": "your_neptune_api_token",
            },
        }
        trainer_with_loggers = DummyTrainer.add_on(WandBLoggerAddon, NeptuneLoggerAddon)
        try:
            trainer = trainer_with_loggers(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                wandb_config=web_logger["wandb"],
                neptune_config=web_logger["neptune"],
            )
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(f"Neptune initialization failed: {e}")
        else:
            self.assertTrue(trainer.neptune_run is not None)

    # ------------------------------------- Legacy ------------------------------------- #
    # def test_result(self):
    #     result_saver = ResultSaver(keys=['loss', 'acc'],
    #                                callback_step=self.step_check,
    #                                callback_epoch=self.epoch_check)
    #     trainer = DummyTrainer(
    #         self.model,
    #         self.optimizer,
    #         self.scheduler,
    #         'train',
    #         callbacks=[result_saver]
    #     )
    #     trainer.run(3, self.train_loader, self.valid_loader)
    #
    # def test_ckpt(self):
    #
    #     ckpt_saver = CKPTSaver('ckpt.pth', callback_step=self.step_check, callback_epoch=self.epoch_check)
    #     trainer = DummyTrainer(
    #         self.model,
    #         self.optimizer,
    #         self.scheduler,
    #         'train',
    #         callbacks=[ckpt_saver]
    #     )
    #     trainer.run(3, self.train_loader, self.valid_loader)
    #     ckpt = torch.load('ckpt.pth')
    #
    #
    # def test_chain(self):
    #     result_saver = ResultSaver(keys=['loss'], callback_epoch=self.epoch_check, callback_step=self.step_check)
    #     ckpt_saver = CKPTSaver('ckpt.pth', callback_epoch=self.epoch_check, callback_step=self.step_check)
    #     trainer = DummyTrainer(
    #         self.model,
    #         self.optimizer,
    #         self.scheduler,
    #         mode='train',
    #         callbacks=[result_saver, ckpt_saver]
    #     )
    #     trainer.run(3, self.train_loader, self.valid_loader)

    # ------------------------------------- Legacy ------------------------------------- #

    # def test_not_notated_epoch_step(self):
    #     with self.assertRaises(ValueError):
    #         CallbackInterface()

    # def test_not_given(self):
    #     trainer = DummyTrainer(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, mode="train")
    #     trainer.run(3, self.train_loader, self.valid_loader)

    # def test_NeptuneCallback(self):
    #     neptune_callback = NeptuneCallback('acc', 'loss', callback_step=self.step_check)
    #     trainer = DummyTrainer(
    #         model=self.model,
    #         optimizer=self.optimizer,
    #         scheduler=self.scheduler,
    #         mode='train',
    #         callbacks=[neptune_callback]
    #     )
    #     trainer.init_tkn(
    #         logging_step=self.step_check,
    #         project_name='CI-test',
    #         api_tkn=os.environ['NEPTUNE_API_TOKEN'],
    #         key='T' + self.test_time_point,
    #     )
    #
    #     trainer.run(3, self.train_loader, self.valid_loader)

    # def test_progress_checker(self):
    #     checker = EpochStepChecker(callback_step=self.step_check, callback_epoch=self.epoch_check)
    #     trainer = DummyTrainer(
    #         model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, mode="train", callbacks=[checker]
    #     )
    #     trainer.run(3, self.train_loader, self.valid_loader)

    # def test_log_stopped(self):
    #     checker = EpochStepChecker(self.step_check, self.epoch_check)
    #     trainer = DummyTrainer(
    #         model=self.model,
    #         optimizer=self.optimizer,
    #         scheduler=self.scheduler,
    #         mode='train',
    #         callbacks=[checker]
    #     )
    #     trainer.init_tkn(project_name='CI-test', api_tkn='API_token')
    #     trainer.run(1, self.train_loader, self.valid_loader)
    #     trainer.web_logger.log('train/acc', 1)
    #     trainer.inference(self.valid_loader)


if __name__ == "__main__":
    unittest.main()
