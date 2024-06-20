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
        self.lower = 0.0
        self.upper = 1.0
        self.activation = ClippedReLU(upper=self.upper, lower=self.lower)
        self.tensors = [
            torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.0, 0.5, 1.0]),
        ]

    def test_clipped_relu_output(self):
        expected_outputs = [
            torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.5]),
            torch.tensor([0.0, 0.5, 1.0]),
        ]
        for tensor, expected in zip(self.tensors, expected_outputs):
            with self.subTest(tensor=tensor):
                output = self.activation(tensor)
                self.assertTrue(torch.allclose(output, expected), f"Failed for input: {tensor}")

    def test_clipped_relu_grad(self):
        for tensor in self.tensors:
            tensor.requires_grad_(True)
            output = self.activation(tensor)
            output.sum().backward()
            grad = tensor.grad
            expected_grad = torch.ones_like(tensor)
            expected_grad[(tensor < self.lower) | (tensor > self.upper)] = 0
            with self.subTest(tensor=tensor):
                self.assertTrue(torch.allclose(grad, expected_grad), f"Gradient check failed for input: {tensor}")

    def test_clipped_relu_invalid_bounds(self):
        with self.assertRaises(AssertionError):
            ClippedReLU(upper=0.0, lower=1.0)

    def test_clipped_relu_lower_bound(self):
        activation = ClippedReLU(upper=2.0, lower=1.0)
        tensor = torch.tensor([0.0, 1.0, 1.5, 2.0, 3.0])
        expected = torch.tensor([1.0, 1.0, 1.5, 2.0, 2.0])
        output = activation(tensor)
        self.assertTrue(torch.allclose(output, expected), f"Failed for input: {tensor}")


if __name__ == "__main__":
    unittest.main()
