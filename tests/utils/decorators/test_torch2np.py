import unittest
import numpy as np
import torch
from zae_engine.utils.decorators.torch2np import torch2np


class TestTorch2NpDecorator(unittest.TestCase):

    def test_torch2np_function(self):
        @torch2np(np.float32, n=2)
        def example_func(x, y, z):
            return x, y, z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        x_np, y_np, z_torch = example_func(x, y, z)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertIsInstance(z_torch, torch.Tensor)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)

    def test_torch2np_method(self):
        class Example:
            @torch2np(np.float32, "x", "y")
            def example_method(self, batch):
                return batch

        example = Example()

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.float32),
            "y": torch.tensor([4, 5, 6], dtype=torch.float32),
            "z": torch.tensor([7, 8, 9], dtype=torch.float32),
        }

        result = example.example_method(batch)

        self.assertIsInstance(result["x"], np.ndarray)
        self.assertIsInstance(result["y"], np.ndarray)
        self.assertIsInstance(result["z"], torch.Tensor)
        self.assertEqual(result["x"].dtype, np.float32)
        self.assertEqual(result["y"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
