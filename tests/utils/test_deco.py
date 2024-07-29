import time
import unittest
import numpy as np
import torch


from zae_engine.utils.deco import np2torch, torch2np, shape_check, tictoc


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


class TestDecorators(unittest.TestCase):

    def setUp(self):
        self.np_array1 = np.array([1.0, 2.0, 3.0])
        self.np_array2 = np.array([4.0, 5.0, 6.0])
        self.torch_tensor1 = torch.tensor([1.0, 2.0, 3.0])
        self.torch_tensor2 = torch.tensor([4.0, 5.0, 6.0])

    @np2torch(torch.float32, "x")
    def func_np2torch(self, batch):
        return batch

    @torch2np(np.float32, "x")
    def func_torch2np(self, batch):
        return batch

    def test_np2torch(self):
        batch = {"x": self.np_array1, "y": self.np_array2}
        result = self.func_np2torch(batch)
        self.assertTrue(isinstance(result["x"], torch.Tensor))
        self.assertTrue(isinstance(result["y"], np.ndarray))
        self.assertTrue(result["x"].dtype, torch.float32)

    def test_torch2np(self):
        batch = {"x": self.torch_tensor1, "y": self.torch_tensor2}
        result = self.func_torch2np(batch)
        self.assertTrue(isinstance(result["x"], np.ndarray))
        self.assertTrue(isinstance(result["y"], torch.Tensor))
        self.assertTrue(result["x"].dtype, np.float32)

    @np2torch(torch.float32, n=2)
    def func_np2torch_positional(self, x, y, z):
        return x, y, z

    @torch2np(np.float32, n=2)
    def func_torch2np_positional(self, x, y, z):
        return x, y, z

    def test_np2torch_positional(self):
        result = self.func_np2torch_positional(self.np_array1, self.np_array2, self.np_array1)
        self.assertTrue(isinstance(result[0], torch.Tensor))
        self.assertTrue(isinstance(result[1], torch.Tensor))
        self.assertTrue(isinstance(result[2], np.ndarray))
        self.assertTrue(result[0].dtype, torch.float32)
        self.assertTrue(result[1].dtype, torch.float32)

    def test_torch2np_positional(self):
        result = self.func_torch2np_positional(self.torch_tensor1, self.torch_tensor2, self.torch_tensor1)
        self.assertTrue(isinstance(result[0], np.ndarray))
        self.assertTrue(isinstance(result[1], np.ndarray))
        self.assertTrue(isinstance(result[2], torch.Tensor))
        self.assertTrue(result[0].dtype, np.float32)
        self.assertTrue(result[1].dtype, np.float32)
