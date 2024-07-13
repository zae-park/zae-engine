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
        pred = np.array([1, 2, 3, 4])
        true = np.array([1, 2, 2, 4])

        expected_f_beta = 0.75
        computed_f_beta = f_beta(pred, true, beta=1.0, num_classes=5, average="micro")
        self.assertAlmostEqual(computed_f_beta.item(), expected_f_beta, places=4)

    def test_f_beta_macro(self):
        pred = np.array([0, 1, 2, 0])
        true = np.array([0, 1, 1, 0])

        expected_f_beta = 0.55555558
        computed_f_beta = f_beta(pred, true, beta=1.0, num_classes=3, average="macro")
        self.assertAlmostEqual(computed_f_beta.item(), expected_f_beta, places=4)

    def test_f_beta_from_mat_micro(self):
        conf_mat = np.array([[5, 2], [1, 3]])

        expected_f_beta = 0.7273
        computed_conf_mat = f_beta_from_mat(conf_mat, beta=1.0, num_classes=2, average="micro")
        self.assertAlmostEqual(computed_conf_mat.item(), expected_f_beta, places=4)

    def test_f_beta_from_mat_macro(self):
        conf_mat = np.array([[5, 2], [1, 3]])

        expected_f_beta = 0.71794867
        computed_conv_mat = f_beta_from_mat(conf_mat, beta=1.0, num_classes=2, average="macro")
        self.assertAlmostEqual(computed_conv_mat.item(), expected_f_beta, places=4)


if __name__ == "__main__":
    unittest.main()
