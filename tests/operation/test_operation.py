import unittest

import torch
import numpy as np

from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix
from zae_engine.operation import MorphologicalLayer, label_to_onoff, onoff_to_label


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


class TestSegmentCodec(unittest.TestCase):
    def test_label_to_onoff_single(self):
        # Test single data sample
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        expected_onoff = [
            [0, 4, 1],  # Label 1 from index 0 to 4
            [5, 6, 2]   # Label 2 from index 5 to 6
        ]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=0)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_multiple(self):
        # Test multiple data samples
        labels = np.array([
            [0, 0, 1, 1, 1, 0, 2, 2],
            [0, 1, 1, 2, 2, 1, 0, 0]
        ])
        expected_onoff = [
            [[0, 4, 1], [5, 6, 2]],
            [[1, 1, 1], [3, 4, 2], [5, 5, 1]]
        ]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=0)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_sensitivity(self):
        # Test sensitivity parameter
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])
        # With sense=3, runs shorter than 3 should be ignored
        expected_onoff = [
            [5, 7, 1]
        ]
        result = label_to_onoff(labels, sense=3, middle_only=False, outside_idx=0)
        self.assertEqual(result, expected_onoff)

    def test_label_to_onoff_middle_only(self):
        # Test middle_only parameter
        labels = np.array([1, 1, 0, 1, 1])
        # middle_only=True should ignore the first and last runs
        expected_onoff = []
        result = label_to_onoff(labels, sense=2, middle_only=True, outside_idx=0)
        self.assertEqual(result, expected_onoff)

    def test_onoff_to_label_single(self):
        # Test single data sample
        onoff = np.array([
            [0, 4, 1],
            [5, 6, 2]
        ])
        expected_labels = np.array([1, 1, 1, 1, 1, 2, 2, 0])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_multiple(self):
        # Test multiple runs
        onoff = np.array([
            [1, 1, 1],
            [3, 4, 2]
        ])
        expected_labels = np.array([0, 1, 0, 2, 2, 0, 0, 0])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_nan(self):
        # Test with np.nan for boundaries
        onoff = np.array([
            [np.nan, 3, 1],
            [4, np.nan, 2]
        ])
        expected_labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

    def test_onoff_to_label_invalid_shape(self):
        # Test invalid onoff shape
        onoff = np.array([0, 4, 1])  # Should be 2D
        with self.assertRaises(IndexError):
            onoff_to_label(onoff, length=8)

    def test_label_to_onoff_invalid_shape(self):
        # Test invalid labels shape
        labels = np.array([])  # Empty array
        with self.assertRaises(IndexError):
            label_to_onoff(labels, sense=2, middle_only=False, outside_idx=0)

    def test_label_to_onoff_tensor_input(self):
        # Test torch.Tensor input
        labels = torch.tensor([0, 0, 1, 1, 1, 0, 2, 2])
        expected_onoff = [
            [0, 4, 1],
            [5, 6, 2]
        ]
        result = label_to_onoff(labels, sense=2, middle_only=False, outside_idx=0)
        self.assertEqual(result, expected_onoff)

    def test_onoff_to_label_tensor_input(self):
        # Test torch.Tensor input
        onoff = torch.tensor([
            [0, 4, 1],
            [5, 6, 2]
        ])
        expected_labels = np.array([1, 1, 1, 1, 1, 2, 2, 0])
        result = onoff_to_label(onoff, length=8)
        np.testing.assert_array_equal(result, expected_labels)

if __name__ == '__main__':
    unittest.main()