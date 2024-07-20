import unittest
import torch
from torch import autograd

from zae_engine.nn_night import GumbelSoftMax


class TestGumbelSoftMax(unittest.TestCase):

    def setUp(self):
        self.tmp = torch.rand(10, dtype=torch.float64).clone().detach().requires_grad_(True)
        self.expected_grad = torch.ones_like(self.tmp)

    def test_forward(self):
        # Test forward pass
        output = GumbelSoftMax.apply(self.tmp)
        rounded = torch.round(self.tmp)
        expected_output = self.tmp + rounded - self.tmp.detach()
        self.assertTrue(torch.equal(output, expected_output))

    def test_backward(self):
        # Test backward pass
        output = GumbelSoftMax.apply(self.tmp)
        output.sum().backward()
        self.assertTrue(torch.equal(self.tmp.grad, self.expected_grad))

    def test_gradcheck(self):
        # Test gradient check
        self.assertTrue(autograd.gradcheck(GumbelSoftMax.apply, (self.tmp,)))


if __name__ == "__main__":
    unittest.main()
