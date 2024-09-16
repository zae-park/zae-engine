import unittest
import numpy as np
import torch
from zae_engine.utils.decorators import torch2np


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

    def test_torch2np_function_all_args(self):
        @torch2np(np.float32)
        def example_func(x, y, z):
            return x, y, z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        x_np, y_np, z_np = example_func(x, y, z)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertIsInstance(z_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)
        self.assertEqual(z_np.dtype, np.float32)

    def test_torch2np_method_all_args(self):
        class Example:
            @torch2np(np.float32)
            def example_method(self, x, y):
                return x, y

        example = Example()

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)

        x_np, y_np = example.example_method(x, y)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)

    def test_torch2np_function_with_non_tensor_args(self):
        @torch2np(np.float32)
        def example_func(x, y, z):
            return x, y, z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = [4, 5, 6]  # Non-tensor argument
        z = "string"  # Non-tensor argument

        x_np, y_out, z_out = example_func(x, y, z)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_out, y)
        self.assertEqual(z_out, z)

    def test_torch2np_method_with_non_tensor_args(self):
        class Example:
            @torch2np(np.float32)
            def example_method(self, x, y):
                return x, y

        example = Example()

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = [4, 5, 6]  # Non-tensor argument

        x_np, y_out = example.example_method(x, y)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_out, y)

    def test_torch2np_static_method(self):
        class Example:
            @staticmethod
            @torch2np(np.float32)
            def example_static_method(x, y):
                return x, y

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)

        x_np, y_np = Example.example_static_method(x, y)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)

    def test_torch2np_class_method(self):
        class Example:
            @classmethod
            @torch2np(np.float32)
            def example_class_method(cls, x, y):
                return x, y

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)

        x_np, y_np = Example.example_class_method(x, y)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)

    def test_torch2np_method_with_missing_keys(self):
        class Example:
            @torch2np(np.float32, "x", "y")
            def example_method(self, batch):
                return batch

        example = Example()

        batch = {
            "a": torch.tensor([1, 2, 3], dtype=torch.float32),
            "b": torch.tensor([4, 5, 6], dtype=torch.float32),
            "c": torch.tensor([7, 8, 9], dtype=torch.float32),
        }

        # "x"와 "y" 키가 존재하지 않으므로 KeyError가 발생해야 함
        with self.assertRaises(KeyError):
            example.example_method(batch)

    def test_torch2np_with_invalid_dtype(self):
        with self.assertRaises(TypeError):

            @torch2np("invalid_dtype")
            def example_func(ex):
                return ex

            x = torch.tensor([1, 2, 3], dtype=torch.float32)
            example_func(x)

    def test_torch2np_with_different_dtype(self):
        @torch2np(np.int32)
        def example_func(x):
            return x

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        x_np = example_func(x)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.int32)

    def test_torch2np_function_n_greater_than_args(self):
        @torch2np(np.float32, n=5)
        def example_func(x, y):
            return x, y

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)

        x_np, y_np = example_func(x, y)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
