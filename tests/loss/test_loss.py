import unittest

import numpy as np
import torch

from zae_engine.loss._loss import cross_entropy, batch_wise_dot


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
        samples = torch.randn(size=(10, 256))
        dot = batch_wise_dot(samples)


if __name__ == "__main__":
    unittest.main()
