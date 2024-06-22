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


class TestClippedReLU_GPT(unittest.TestCase):
    def setUp(self):
        self.clipped_relu_default = ClippedReLU()
        self.clipped_relu_custom = ClippedReLU(upper=2.0, lower=-1.0)

    def test_default_thresholds(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
        expected_output = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        output = self.clipped_relu_default(x)
        self.assertTrue(torch.equal(output, expected_output))

    def test_custom_thresholds(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
        expected_output = torch.tensor([-1.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
        output = self.clipped_relu_custom(x)
        self.assertTrue(torch.equal(output, expected_output))

    def test_gradient(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0], requires_grad=True)
        output = self.clipped_relu_default(x)
        output.sum().backward()
        expected_grad = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_invalid_thresholds(self):
        with self.assertRaises(AssertionError):
            ClippedReLU(upper=0.0, lower=1.0)

    def test_forward_shape(self):
        x = torch.randn(10, 20)
        output = self.clipped_relu_default(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_dtype(self):
        x = torch.randn(10, 20, dtype=torch.float32)
        output = self.clipped_relu_default(x)
        self.assertEqual(output.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
