import unittest

import numpy as np

from zae_engine.metrics.utils import np2torch, shape_check


class TestDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self.x_ex = np.array([0] * 500 + [1] * 500 + [0] * 1500)
        self.y_ex = np.array([0] * 600 + [1] * 400 + [0] * 1500)

    def test_np2torch(self):
        pass

    def test_shape_check(self):
        pass


if __name__ == "__main__":
    unittest.main()
