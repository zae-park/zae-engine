import unittest

import torch
import numpy as np
from random import randint, choice

from zae_engine.loss import cross_entropy, compute_gram_matrix
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


class TestComputeGram(unittest.TestCase):

    def test_gram_mat(self):
        batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        norm1 = torch.norm(batch[0])
        norm2 = torch.norm(batch[1])

        expected_output = torch.tensor(
            [(1.0 + (1 * 3 + 2 * 4) / (norm1 * norm2) + (1 * 3 + 2 * 4) / (norm1 * norm2) + 1.0) / 4],
            dtype=torch.float32,
        )

        output = compute_gram_matrix(batch)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_gram_mat_reduce_false(self):
        batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        norm1 = torch.norm(batch[0])
        norm2 = torch.norm(batch[1])
        expected_output = torch.tensor(
            [[1.0, (1 * 3 + 2 * 4) / (norm1 * norm2)], [(1 * 3 + 2 * 4) / (norm1 * norm2), 1.0]], dtype=torch.float32
        )

        output = compute_gram_matrix(batch, reduce=False)
        for o, e in zip(output.view(-1).tolist(), expected_output.view(-1).tolist()):
            self.assertAlmostEqual(o, e, places=4)

    def test_gram_mat_with_different_vectors(self):
        batch = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        expected_output = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        output = compute_gram_matrix(batch)
        self.assertTrue(torch.allclose(output, expected_output.mean(), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
