import unittest
import torch
from torch import autograd

from zae_engine.nn_night import GumbelSoftMax


class TestGumbelSoftMax(unittest.TestCase):

    def setUp(self):
        self.input = torch.rand(10, dtype=torch.float64, requires_grad=True)

    def test_forward(self):
        output = GumbelSoftMax.apply(self.input)
        self.assertEqual(output.shape, self.input.shape)

    def test_backward(self):
        output = GumbelSoftMax.apply(self.input)
        output.backward(torch.ones_like(self.input))
        self.assertIsNotNone(self.input.grad)


if __name__ == "__main__":
    unittest.main()
