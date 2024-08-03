import time
import unittest
import numpy as np
import torch

from zae_engine.utils.decorators import np2torch, torch2np, shape_check, tictoc


@np2torch(torch.float32, n=2)
def example_np2torch(x, y, z):
    return x, y, z


@torch2np(np.float32, n=2)
def example_torch2np(x, y, z):
    return x, y, z


@shape_check(2)
def example_shape_check_1(x, y):
    return x + y


@shape_check("x", "y")
def example_shape_check_2(x=None, y=None):
    return x + y


@tictoc
def example_tictoc():
    time.sleep(1)


class TestDecorators(unittest.TestCase):

    def test_np2torch_all(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        x_torch, y_torch, z_np = example_np2torch(x, y, z)

        self.assertIsInstance(x_torch, torch.Tensor)
        self.assertIsInstance(y_torch, torch.Tensor)
        self.assertIsInstance(z_np, np.ndarray)
        self.assertEqual(x_torch.dtype, torch.float32)
        self.assertEqual(y_torch.dtype, torch.float32)

    def test_torch2np_all(self):
        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        x_np, y_np, z_torch = example_torch2np(x, y, z)

        self.assertIsInstance(x_np, np.ndarray)
        self.assertIsInstance(y_np, np.ndarray)
        self.assertIsInstance(z_torch, torch.Tensor)
        self.assertEqual(x_np.dtype, np.float32)
        self.assertEqual(y_np.dtype, np.float32)

    def test_shape_check_positional(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self.assertEqual(example_shape_check_1(x, y).tolist(), [5, 7, 9])

    def test_shape_check_keyword(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self.assertEqual(example_shape_check_2(x=x, y=y).tolist(), [5, 7, 9])

    def test_shape_check_failure(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([[4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError):
            example_shape_check_1(x, y)

    def test_tictoc(self):
        start = time.time()
        example_tictoc()
        end = time.time()
        self.assertTrue(end - start >= 1)


class TestDecoratorsWithDictAndTuple(unittest.TestCase):

    def setUp(self):
        self.dict_data = {
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6]),
            "z": torch.tensor([7, 8, 9], dtype=torch.float32),
        }
        self.torch_data = (
            torch.tensor([1, 2, 3], dtype=torch.float32),
            torch.tensor([4, 5, 6], dtype=torch.float32),
            torch.tensor([7, 8, 9], dtype=torch.float32),
        )
        self.numpy_data = (
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
            np.array([7, 8, 9], dtype=np.float32),
        )

    @np2torch(torch.float32, "x", "y")
    def sample_func_np2torch_dict(self, batch):
        return batch

    @torch2np(np.float32, "x", "y")
    def sample_func_torch2np_dict(self, batch):
        return batch

    @np2torch(torch.float32, n=2)
    def sample_func_np2torch_tuple(self, x, y, z):
        return x, y, z

    @torch2np(np.float32, n=2)
    def sample_func_torch2np_tuple(self, x, y, z):
        return x, y, z

    def test_np2torch_dict(self):
        result = self.sample_func_np2torch_dict(self.dict_data)
        self.assertIsInstance(result["x"], torch.Tensor)
        self.assertIsInstance(result["y"], torch.Tensor)
        self.assertIsInstance(result["z"], torch.Tensor)

    def test_torch2np_dict(self):
        result = self.sample_func_torch2np_dict(self.dict_data)
        self.assertIsInstance(result["x"], np.ndarray)
        self.assertIsInstance(result["y"], np.ndarray)
        self.assertIsInstance(result["z"], torch.Tensor)

    def test_np2torch_tuple(self):
        x, y, z = self.sample_func_np2torch_tuple(*self.numpy_data)
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(z, np.ndarray)

    def test_torch2np_tuple(self):
        x, y, z = self.sample_func_torch2np_tuple(*self.torch_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(z, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
