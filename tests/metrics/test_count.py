import unittest

import numpy as np

from zae_engine.metrics.count import accuracy, f_beta, f_beta_from_mat


class TestCount(unittest.TestCase):
    def setUp(self) -> None:
        self.x_ex = np.array([0] * 500 + [1] * 500 + [0] * 1500)
        self.y_ex = np.array([0] * 600 + [1] * 400 + [0] * 1500)

    def test_accuracy(self):
        self.assertEqual(accuracy(self.y_ex, self.x_ex), 24 / 25)

    def test_fbeta(self):
        pass


if __name__ == "__main__":
    unittest.main()
