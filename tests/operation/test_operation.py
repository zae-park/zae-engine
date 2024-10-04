import unittest

import torch
import numpy as np

from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix
from zae_engine.operation import MorphologicalLayer, arg_nearest


class TestMorphology(unittest.TestCase):
    def setUp(self) -> None:
        self.zit_tensor = torch.tensor(np.array([1] * 80 + [0, 1] * 20 + [0] * 80), dtype=torch.float32).reshape(
            1, 1, -1
        )
        self.open_close = MorphologicalLayer("CO", [9, 9])
        self.close_open = MorphologicalLayer("OC", [9, 9])
        super().setUp()

    def test_morph(self):
        res1 = self.open_close(self.zit_tensor)
        res2 = self.close_open(self.zit_tensor)

        self.assertEqual(res1.shape, res2.shape)
        self.assertGreaterEqual(res1.sum(), res2.sum())


class TestDrawConfusion(unittest.TestCase):

    def setUp(self) -> None:
        self.num_classes = 30
        self.yhat_ex = np.random.randint(0, self.num_classes, 2000)
        self.y_ex = np.random.randint(0, self.num_classes, 2000)
        self.no_elements = np.array([])
        return super().setUp()

    def check_shape_and_classes(self):
        cm = confusion_matrix(self.y_ex, self.yhat_ex, num_classes=self.num_classes)
        classes, _ = cm.shape

        self.assertEqual(classes, self.num_classes)
        self.assertEqual((self.num_classes, self.num_classes), cm.shape)

    def test_zero_num_classes(self):
        with self.assertRaises(Exception):
            confusion_matrix(self.y_ex, self.yhat_ex, num_classes=0)


class TestPrintConfusion(unittest.TestCase):

    def check_input_as_ndarray(self):
        with self.assertRaises(Exception):
            print_confusion_matrix(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))


class TestArgNearest(unittest.TestCase):
    def test_single_nearest_with_value(self):
        arr = np.array([1, 3, 5, 7, 9])
        value = 6
        expected = (2, 5)
        result = arg_nearest(arr, value)
        self.assertEqual(result, expected)

    def test_single_nearest_without_value(self):
        arr = np.array([1, 3, 5, 7, 9])
        value = 6
        expected = 2
        result = arg_nearest(arr, value, return_value=False)
        self.assertEqual(result, expected)

    def test_min_value(self):
        arr = np.array([10, 20, 30, 40, 50])
        value = 5
        expected = (0, 10)
        result = arg_nearest(arr, value)
        self.assertEqual(result, expected)

    def test_max_value(self):
        arr = np.array([2, 4, 6, 8, 10])
        value = 12
        expected = (4, 10)
        result = arg_nearest(arr, value)
        self.assertEqual(result, expected)

    def test_exact_match(self):
        arr = np.array([1, 2, 3, 4, 5])
        value = 3
        expected = (2, 3)
        result = arg_nearest(arr, value)
        self.assertEqual(result, expected)

    def test_torch_tensor_input(self):
        arr_tensor = torch.tensor([2, 4, 6, 8, 10])
        value = 7
        expected = (2, 6)
        result = arg_nearest(arr_tensor, value)
        self.assertEqual(result, expected)

    def test_unsorted_array(self):
        arr = np.array([1, 5, 3, 7, 9])
        value = 4
        with self.assertRaises(ValueError):
            arg_nearest(arr, value)

    def test_multidimensional_array(self):
        arr = np.array([[1, 2], [3, 4]])
        value = 3
        with self.assertRaises(ValueError):
            arg_nearest(arr, value)

    def test_invalid_input_type(self):
        arr = [1, 2, 3, 4, 5]  # Not a NumPy array or PyTorch tensor
        value = 3
        with self.assertRaises(TypeError):
            arg_nearest(arr, value)

    def test_return_value_parameter(self):
        arr = np.array([1, 2, 3, 4, 5])
        value = 4
        expected_with_value = (3, 4)
        expected_without_value = 3
        result_with_value = arg_nearest(arr, value, return_value=True)
        result_without_value = arg_nearest(arr, value, return_value=False)
        self.assertEqual(result_with_value, expected_with_value)
        self.assertEqual(result_without_value, expected_without_value)


if __name__ == "__main__":
    unittest.main()
