import unittest

import numpy as np
import torch

from zae_engine.loss._loss import cross_entropy, batch_wise_dot


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.attr_dict = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_cross_entropy(self):
        self.cls_tuple = self.attr_dict["cls"]
        self.seg_tuple = self.attr_dict["seg"]
        loss = _loss.cross_entropy(self.cls_tuple[0], self.cls_tuple[-1])
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.size().numel(), 1)
        self.assertAlmostEqual(float(loss), 0.80846738, places=4)

        loss = _loss.cross_entropy(self.seg_tuple[0], self.seg_tuple[-1])
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.size().numel(), 1)
        self.assertAlmostEqual(float(loss), 0.80567473, places=4)

    def test_batch_dot(self):
        samples = torch.randn(size=(10, 256))
        dot = batch_wise_dot(samples)


if __name__ == "__main__":
    unittest.main()
