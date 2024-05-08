import unittest

import numpy as np
import torch
from random import randint, random

from zae_engine.models import DummyModel
from zae_engine.utils import scheduler, EPS


class TestScheduler(unittest.TestCase):
    model = None
    optimizer = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DummyModel()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.total_step = randint(0, 1024)
        a, b = random(), random()
        self.eta_min, self.eta_max = min(a, b), max(a, b)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eta_max)

    def tearDown(self) -> None:
        pass

    def sweep_lrs(self, schedule: torch.optim.lr_scheduler) -> list:

        lrs = []
        for _ in range(self.total_step):
            lrs.append(self.optimizer.param_groups[0]["lr"])
            schedule.step()
        return lrs

    def test_warmup(self):
        schedule = scheduler.WarmUpScheduler(self.optimizer, self.total_step, eta_min=self.eta_min)
        lrs = self.sweep_lrs(schedule)

        self.assertEqual(lrs[-1], self.eta_max)
        # self.assertEqual(lrs[0], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs) > 0))
        self.assertLessEqual(np.mean(np.diff(np.diff(lrs))), EPS)

    def test_cosine_annealing(self):
        schedule = scheduler.CosineAnnealingScheduler(self.optimizer, self.total_step, eta_min=self.eta_max)
        lrs = self.sweep_lrs(schedule)

        self.assertEqual(lrs[0], self.eta_max)
        # self.assertEqual(lrs[-1], self.eta_min)
        self.assertFalse(np.all(np.diff(lrs) < 0))

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
