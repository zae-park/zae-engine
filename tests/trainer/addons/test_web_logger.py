import os
import shutil
import unittest
import gc
from typing import Union, Dict
from datetime import datetime

import wandb
from neptune.exceptions import NeptuneInvalidApiTokenException
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


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
        wandb.setup(wandb.Settings(program="test_callback.py", program_relpath="test_callback.py"))
        try:
            self.runner = wandb.init(project="wandb-test", config={"a": 1, "b": 2, "c": 3})
        except wandb.errors.UsageError as e:
            self.skipTest(e)

    def tearDown(self) -> None:
        self.step_check = None
        self.epoch_check = None
        self.test_time_point = None
        if self.runner:
            self.runner.finish()
        self.runner = None

    def test_wandb_init(self):
        self.assertIn("wandb", os.listdir(".."))

    def test_wandb_log(self):
        self.runner.log({"test": True})
        self.assertTrue(self.runner.summary["test"])

    def test_neptune_init(self):
        web_logger = {
            "neptune": {
                "project_name": "test_project",
                "api_tkn": "your_neptune_api_token",
            }
        }
        trainer_with_neptune = DummyTrainer.add_on(NeptuneLoggerAddon)
        try:
            trainer = trainer_with_neptune(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                web_logger=web_logger,
            )
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(e)
        else:
            self.assertTrue(trainer.web_logger["neptune"].is_live())

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
                web_logger=web_logger,
            )
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(e)
        else:
            self.assertTrue(trainer.web_logger["neptune"].is_live())
            self.assertTrue(trainer.web_logger["wandb"].is_live())

    def test_wandb_logging(self):
        web_logger = {"wandb": {"project": "wandb-test", "config": {"a": 1, "b": 2, "c": 3}}}
        trainer_with_wandb = DummyTrainer.add_on(WandBLoggerAddon)
        trainer = trainer_with_wandb(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            web_logger=web_logger,
        )
        trainer.log_train["loss"] = [0.1]
        trainer.logging({"loss": torch.tensor(0.2)})
        self.assertIn("loss", trainer.log_train)

    def test_neptune_logging(self):
        web_logger = {"neptune": {"project_name": "test_project", "api_tkn": "your_neptune_api_token"}}
        trainer_with_neptune = DummyTrainer.add_on(NeptuneLoggerAddon)
        try:
            trainer = trainer_with_neptune(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                web_logger=web_logger,
            )
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(e)
        else:
            trainer.log_train["loss"] = [0.1]
            trainer.logging({"loss": torch.tensor(0.2)})
            self.assertIn("loss", trainer.log_train)

    def test_wandb_del(self):
        web_logger = {"wandb": {"project": "wandb-test", "config": {"a": 1, "b": 2, "c": 3}}}
        trainer_with_wandb = DummyTrainer.add_on(WandBLoggerAddon)
        trainer = trainer_with_wandb(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            web_logger=web_logger,
        )
        trainer_id = id(trainer)
        del trainer
        self.assertNotIn(trainer_id, [id(obj) for obj in gc.get_objects()])

    def test_neptune_del(self):
        web_logger = {"neptune": {"project_name": "test_project", "api_tkn": "your_neptune_api_token"}}
        trainer_with_neptune = DummyTrainer.add_on(NeptuneLoggerAddon)
        try:
            trainer = trainer_with_neptune(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                web_logger=web_logger,
            )
        except NeptuneInvalidApiTokenException as e:
            self.skipTest(e)
        else:
            trainer_id = id(trainer)
            del trainer
            self.assertNotIn(trainer_id, [id(obj) for obj in gc.get_objects()])


if __name__ == "__main__":
    unittest.main()
