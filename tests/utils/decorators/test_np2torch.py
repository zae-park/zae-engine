import unittest
import numpy as np
import torch
from zae_engine.utils.decorators import np2torch


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

        batch = {
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6]),
            "z": np.array([7, 8, 9]),
        }

        result = example.example_method(batch)

        self.assertIsInstance(result["x"], torch.Tensor)
        self.assertIsInstance(result["y"], torch.Tensor)
        self.assertIsInstance(result["z"], np.ndarray)
        self.assertEqual(result["x"].dtype, torch.float32)
        self.assertEqual(result["y"].dtype, torch.float32)

    def test_np2torch_function_all_args(self):
        @np2torch(torch.float32)
        def example_func(x, y, z):
            return x, y, z

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        x_torch, y_torch, z_torch = example_func(x, y, z)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertIsInstance(z_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)
        self.assertEqual(z_torch.dtype, torch.float32)

    def test_np2torch_method_all_args(self):
        class Example:
            @np2torch(torch.float32)
            def example_method(self, x, y):
                return x, y

        example = Example()

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        x_torch, y_torch = example.example_method(x, y)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)

    def test_np2torch_function_with_non_array_args(self):
        @np2torch(torch.float32)
        def example_func(x, y, z):
            return x, y, z

        x = np.array([1, 2, 3])
        y = [4, 5, 6]  # Non-numpy array argument
        z = "string"  # Non-numpy array argument

        x_torch, y_out, z_out = example_func(x, y, z)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_out, y)
        self.assertEqual(z_out, z)

    def test_np2torch_method_with_non_array_args(self):
        class Example:
            @np2torch(torch.float32)
            def example_method(self, x, y):
                return x, y

        example = Example()

        x = np.array([1, 2, 3])
        y = [4, 5, 6]  # Non-numpy array argument

        x_torch, y_out = example.example_method(x, y)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_out, y)

    def test_np2torch_static_method(self):
        class Example:
            @staticmethod
            @np2torch(torch.float32)
            def example_static_method(x, y):
                return x, y

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        x_torch, y_torch = Example.example_static_method(x, y)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)

    def test_np2torch_class_method(self):
        class Example:
            @classmethod
            @np2torch(torch.float32)
            def example_class_method(cls, x, y):
                return x, y

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        x_torch, y_torch = Example.example_class_method(x, y)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)

    def test_np2torch_method_with_missing_keys(self):
        class Example:
            @np2torch(torch.float32, "x", "y")
            def example_method(self, batch):
                return batch

        example = Example()

        batch = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
            "c": np.array([7, 8, 9]),
        }

        # "x"와 "y" 키가 존재하지 않으므로 KeyError가 발생해야 함
        with self.assertRaises(KeyError):
            example.example_method(batch)

    def test_np2torch_with_invalid_dtype(self):
        with self.assertRaises(TypeError):

            @np2torch("invalid_dtype")
            def example_func(x):
                return x

            x = np.array([1, 2, 3])
            example_func(x)

    def test_np2torch_with_different_dtype(self):
        @np2torch(torch.int32)
        def example_func(x):
            return x

        x = np.array([1, 2, 3], dtype=np.float32)
        x_torch = example_func(x)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.int32)

    def test_np2torch_function_n_greater_than_args(self):
        @np2torch(torch.float32, n=5)
        def example_func(x, y):
            return x, y

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        x_torch, y_torch = example_func(x, y)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
