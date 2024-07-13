import unittest
import numpy as np
import torch

from zae_engine.metrics.count import accuracy, f_beta, f_beta_from_mat


class TestCountMetrics(unittest.TestCase):

    def test_accuracy(self):
        true_np = np.array([1, 2, 3, 4])
        predict_np = np.array([1, 2, 2, 4])
        true_torch = torch.tensor([1, 2, 3, 4])
        predict_torch = torch.tensor([1, 2, 2, 4])

        expected_accuracy = 0.75

        self.assertAlmostEqual(accuracy(true_np, predict_np).item(), expected_accuracy, places=4)
        self.assertAlmostEqual(accuracy(true_torch, predict_torch).item(), expected_accuracy, places=4)

    def test_f_beta_micro(self):
        pred_np = np.array([1, 2, 3, 4])
        true_np = np.array([1, 2, 2, 4])
        pred_torch = torch.tensor([1, 2, 3, 4])
        true_torch = torch.tensor([1, 2, 2, 4])

        expected_f_beta = 0.8

        self.assertAlmostEqual(
            f_beta(pred_np, true_np, beta=1.0, num_classes=5, average="micro").item(), expected_f_beta, places=4
        )
        self.assertAlmostEqual(
            f_beta(pred_torch, true_torch, beta=1.0, num_classes=5, average="micro").item(), expected_f_beta, places=4
        )

    def test_f_beta_macro(self):
        pred_np = np.array([0, 1, 2, 0])
        true_np = np.array([0, 1, 1, 0])
        pred_torch = torch.tensor([0, 1, 2, 0])
        true_torch = torch.tensor([0, 1, 1, 0])

        expected_f_beta = 0.5

        self.assertAlmostEqual(
            f_beta(pred_np, true_np, beta=1.0, num_classes=3, average="macro").item(), expected_f_beta, places=4
        )
        self.assertAlmostEqual(
            f_beta(pred_torch, true_torch, beta=1.0, num_classes=3, average="macro").item(), expected_f_beta, places=4
        )

    def test_f_beta_from_mat_micro(self):
        conf_mat_np = np.array([[5, 2], [1, 3]])
        conf_mat_torch = torch.tensor([[5, 2], [1, 3]])

        expected_f_beta = 0.7273

        self.assertAlmostEqual(
            f_beta_from_mat(conf_mat_np, beta=1.0, num_classes=2, average="micro").item(), expected_f_beta, places=4
        )
        self.assertAlmostEqual(
            f_beta_from_mat(conf_mat_torch, beta=1.0, num_classes=2, average="micro").item(), expected_f_beta, places=4
        )

    def test_f_beta_from_mat_macro(self):
        conf_mat_np = np.array([[5, 2], [1, 3]])
        conf_mat_torch = torch.tensor([[5, 2], [1, 3]])

        expected_f_beta = 0.7083

        self.assertAlmostEqual(
            f_beta_from_mat(conf_mat_np, beta=1.0, num_classes=2, average="macro").item(), expected_f_beta, places=4
        )
        self.assertAlmostEqual(
            f_beta_from_mat(conf_mat_torch, beta=1.0, num_classes=2, average="macro").item(), expected_f_beta, places=4
        )


if __name__ == "__main__":
    unittest.main()
