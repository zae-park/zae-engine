from random import randint, random as rand

import unittest

import torch

from zae_engine.nn_night import ClippedReLU


class TestClippedReLU(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.eps = torch.finfo(torch.float32).eps

    def setUp(self) -> None:
        self.sample = torch.randn(size=(1, 1), requires_grad=True)

    def tearDown(self) -> None:
        pass

    def test_build(self):
        multiply = randint(1, 8)
        a, b = rand() * multiply, rand() * multiply
        random_top, random_bot = max(a, b), min(a, b)

        activation = ClippedReLU(upper=random_top, lower=random_bot)
        out = activation(self.sample * multiply)
        self.assertLessEqual(random_bot, out * 1.001)  # margin for knee in ReLU
        self.assertGreaterEqual(random_top, out * 0.999)  # margin for knee in ReLU

        with self.assertRaises(AssertionError):
            activation = ClippedReLU(upper=random_bot, lower=random_top)  # Error when lower > upper

    def test_clip(self):
        activation = ClippedReLU(rand() + 1, rand())
        out = activation(self.sample)

        self.assertTrue(out.requires_grad)
        out.backward()


if __name__ == "__main__":
    unittest.main()
