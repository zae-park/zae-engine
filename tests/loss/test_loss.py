import unittest

import torch
import numpy as np
from random import randint, choice

from zae_engine.loss._loss import cross_entropy, batch_wise_dot
from zae_engine import utils


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass
        # self.attr_dict = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_cross_entropy(self):
        true = torch.randint(0, 1, size=(1, 128))
        pred = torch.randint(0, 1, size=(1, 128))

        with self.assertRaises(RuntimeError):
            loss = cross_entropy(pred, true)
        # self.assertIsInstance(loss, torch.Tensor)
        # self.assertEqual(loss.size().numel(), 1)
        # self.assertAlmostEqual(float(loss), 0.80846738, places=4)

        # loss = cross_entropy(self.seg_tuple[0], self.seg_tuple[-1])
        # self.assertIsInstance(loss, torch.Tensor)
        # self.assertEqual(loss.size().numel(), 1)
        # self.assertAlmostEqual(float(loss), 0.80567473, places=4)

    def test_batch_dot(self):
        feat_len = randint(1, 512)
        samples = torch.randn(size=(10, feat_len))
        dot_mat = batch_wise_dot(samples, reduce=False)
        self.assertLessEqual((1 - torch.diag(dot_mat)).mean(), utils.EPS * feat_len)
        self.assertLessEqual((torch.transpose(dot_mat, 0, 1) - dot_mat).mean(), utils.EPS)


if __name__ == "__main__":
    unittest.main()
