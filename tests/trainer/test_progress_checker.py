import unittest
from random import randint

from zae_engine.trainer import ProgressChecker


class TestProgressChecker(unittest.TestCase):
    checker = ProgressChecker()

    def setUp(self) -> None:
        self.checker.init_state()

    def test_init_state(self):
        self.assertEqual(self.checker.get_step(), 1)
        self.assertEqual(self.checker.get_epoch(), 1)

    def test_epoch_update(self):
        epoch_update = randint(1, 1024)
        for e in range(epoch_update):
            self.checker.update_epoch()
        self.assertEqual(self.checker.get_epoch(), 1 + epoch_update)

    def test_step_update(self):
        step_update = randint(1, 1024)
        for e in range(step_update):
            self.checker.update_step()
        self.assertEqual(self.checker.get_step(), 1 + step_update)


if __name__ == "__main__":
    unittest.main()
