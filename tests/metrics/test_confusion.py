import unittest
from unittest.mock import patch
import numpy as np
import torch
from io import StringIO

from zae_engine.metrics.confusion import confusion_matrix, print_confusion_matrix


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true_np = np.array([0, 1, 2, 2, 1])
        self.y_hat_np = np.array([0, 2, 2, 2, 0])
        self.y_true_torch = torch.tensor([0, 1, 2, 2, 1])
        self.y_hat_torch = torch.tensor([0, 2, 2, 2, 0])
        self.num_classes = 3
        self.expected_conf_mat = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
        self.class_names = ["Class 0", "Class 1", "Class 2"]

    def test_confusion_matrix_numpy(self):
        conf_mat = confusion_matrix(self.y_hat_np, self.y_true_np, self.num_classes)
        self.assertTrue(torch.equal(conf_mat, self.expected_conf_mat))

    def test_confusion_matrix_torch(self):
        conf_mat = confusion_matrix(self.y_hat_torch, self.y_true_torch, self.num_classes)
        self.assertTrue(torch.equal(conf_mat, self.expected_conf_mat))

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_confusion_matrix_numpy(self, mock_stdout):
        print_confusion_matrix(self.expected_conf_mat.numpy(), class_name=self.class_names)
        output = mock_stdout.getvalue()
        self.assertIn("Class 0", output)
        self.assertIn("Class 1", output)
        self.assertIn("Class 2", output)
        self.assertIn("5", output)
        self.assertIn("2", output)
        self.assertIn("1", output)
        self.assertIn("3", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_confusion_matrix_torch(self, mock_stdout):
        print_confusion_matrix(self.expected_conf_mat, class_name=self.class_names)
        output = mock_stdout.getvalue()
        self.assertIn("Class 0", output)
        self.assertIn("Class 1", output)
        self.assertIn("Class 2", output)
        self.assertIn("5", output)
        self.assertIn("2", output)
        self.assertIn("1", output)
        self.assertIn("3", output)


if __name__ == "__main__":
    unittest.main()
