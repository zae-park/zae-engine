from random import randint, random as rand

import unittest

import torch

from zae_engine.nn_night import ClippedReLU


class TestClippedReLU(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.sample = torch.randn(size=(1, 1), requires_grad=True)

    def tearDown(self) -> None:
        pass

    def test_build(self):
        multiply = randint(1, 8)
        random_top = rand() * multiply
        random_bot = rand() * multiply

        if random_top > random_bot:
            activation = ClippedReLU(random_top, random_bot)
            out = activation(self.sample * multiply)
            self.assertLessEqual(random_bot, out)
            self.assertGreaterEqual(random_top, out)
        else:
            with self.assertRaises(AssertionError):
                activation = ClippedReLU(random_top, random_bot)

    def test_clip(self):
        activation = ClippedReLU(rand() + 1, rand())
        out = activation(self.sample)

        self.assertTrue(out.requires_grad)
        out.backward()


if __name__ == "__main__":
    unittest.main()
