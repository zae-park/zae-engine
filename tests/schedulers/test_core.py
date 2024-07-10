import unittest

import numpy as np
import torch
from random import randint, random

from zae_engine.models import DummyModel
from zae_engine.utils import EPS
from zae_engine.schedulers import scheduler, core


class TestScheduler(unittest.TestCase):
    model = None
    optimizer = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DummyModel()
        a, b = random(), random()
        cls.eta_min, cls.eta_max = min(a, b), max(a, b)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eta_max)

    def tearDown(self) -> None:
        pass

    def sweep_lrs(self, schedule: torch.optim.lr_scheduler) -> list:

        lrs = []
        for _ in range(schedule.total_iters):
            lrs.append(self.optimizer.param_groups[0]["lr"])
            schedule.step()
        return lrs

    def test_base(self):

        with self.assertRaises(TypeError):
            base = core.SchedulerBase(self.optimizer, total_iters=randint(0, 1024), eta_min=self.eta_min)

    def test_chain(self):
        schedule0 = scheduler.WarmUpScheduler(self.optimizer, step1 := randint(1, 1024), eta_min=self.eta_min)
        schedule1 = scheduler.CosineAnnealingScheduler(self.optimizer, step2 := randint(1, 1024), eta_min=self.eta_min)
        schedule2 = scheduler.CosineAnnealingScheduler(self.optimizer, step3 := randint(1, 1024), eta_min=self.eta_min)
        chains = core.SchedulerChain(schedule0, schedule1, schedule2)
        lrs = self.sweep_lrs(chains)

        self.assertEqual(len(lrs), sum([step1, step2, step3]))

        # WarmUp
        self.assertLessEqual(lrs[chains.next_iters[0] - 2], self.eta_max + EPS)
        self.assertGreaterEqual(lrs[1], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs[: chains.next_iters[0]]) > 0))
        self.assertLessEqual(np.mean(np.diff(np.diff(lrs[: chains.next_iters[0]]))), EPS)

        # Cosine 1
        self.assertLessEqual(lrs[chains.next_iters[0]], self.eta_max)
        self.assertGreaterEqual(lrs[chains.next_iters[1] - 1], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs[step1 : chains.next_iters[1] - 1]) < 0))

        # Cosine 2
        self.assertLessEqual(lrs[chains.next_iters[1]], self.eta_max)
        self.assertGreaterEqual(lrs[-1], self.eta_min)
        self.assertTrue(np.all(np.diff(lrs[chains.next_iters[1] : -1]) < 0))


if __name__ == "__main__":
    unittest.main()
