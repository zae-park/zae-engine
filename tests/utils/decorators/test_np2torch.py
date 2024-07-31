import unittest
import numpy as np
import torch
from zae_engine.utils.decorators.np2torch import np2torch


class TestNp2TorchDecorator(unittest.TestCase):

    def test_np2torch_function(self):
        @np2torch(torch.float32, n=2)
        def example_func(x, y, z):
            return x, y, z

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        x_torch, y_torch, z_np = example_func(x, y, z)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertIsInstance(z_np, np.ndarray)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)

    def test_np2torch_method(self):
        class Example:
            @np2torch(torch.float32, "x", "y")
            def example_method(self, batch):
                return batch

        example = Example()

        batch = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])}

        result = example.example_method(batch)

        self.assertIsInstance(result["x"], torch.Tensor)
        self.assertIsInstance(result["y"], torch.Tensor)
        self.assertIsInstance(result["z"], np.ndarray)
        self.assertEqual(result["x"].dtype, torch.float32)
        self.assertEqual(result["y"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
