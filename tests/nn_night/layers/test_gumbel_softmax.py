import unittest
import torch
from torch import autograd

from zae_engine.nn_night import GumbelSoftMax


class TestGumbelSoftMax(unittest.TestCase):

    def setUp(self):
        self.logits = torch.rand(10, 3, dtype=torch.float64).clone().detach().requires_grad_(True)
        self.temperature = 1.0

    def test_forward(self):
        # Test forward pass
        output = GumbelSoftMax.apply(self.logits, self.temperature)
        self.assertEqual(output.shape, self.logits.shape)

    def test_backward(self):
        # Test backward pass
        output = GumbelSoftMax.apply(self.logits, self.temperature)
        output.sum().backward()
        self.assertIsNotNone(self.logits.grad)

    def test_gradcheck(self):
        # Test gradient check
        self.assertTrue(autograd.gradcheck(GumbelSoftMax.apply, (self.logits, self.temperature)))


if __name__ == "__main__":
    unittest.main()
