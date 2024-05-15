import unittest

import numpy as np
import torch
from random import randint, random

from zae_engine.models import DummyModel
from zae_engine.utils import EPS
from zae_engine.schedulers import scheduler


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
        self.total_iters = randint(0, 1024)
        a, b = random(), random()
        self.eta_min, self.eta_max = min(a, b), max(a, b)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eta_max)

    def tearDown(self) -> None:
        pass

    def sweep_lrs(self, schedule: torch.optim.lr_scheduler) -> list:

        lrs = []
        for _ in range(schedule.total_iters):
            lrs.append(self.optimizer.param_groups[0]["lr"])
            schedule.step()
        return lrs

    def test_warmup(self):
        schedule = scheduler.WarmUpScheduler(self.optimizer, self.total_iters, eta_min=self.eta_min)
        lrs = self.sweep_lrs(schedule)

        self.assertLessEqual(lrs[-1], self.eta_max + EPS)
        self.assertGreaterEqual(lrs[0], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs) > 0))
        self.assertLessEqual(np.mean(np.diff(np.diff(lrs))), EPS)

    def test_cosine_annealing(self):
        schedule = scheduler.CosineAnnealingScheduler(self.optimizer, self.total_iters, eta_min=self.eta_min)
        lrs = self.sweep_lrs(schedule)

        self.assertLessEqual(lrs[0], self.eta_max)
        self.assertGreaterEqual(lrs[-1], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs) < 0))


if __name__ == "__main__":
    unittest.main()
