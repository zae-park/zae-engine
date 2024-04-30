from random import randint

import unittest

import torch
import numpy as np

from zae_engine.nn_night import Inv1d


class TestInv1d(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_shape(self):
        random_ch = randint(1, 32)
        random_dim = randint(3, 2048)
        sample = torch.zeros(1, random_ch, random_dim)
        inv_layer = Inv1d(random_ch, num_groups=2, kernel_size=3, stride=1, reduction_ratio=3)
        out = inv_layer(sample)
        kernel_size, stride = inv_layer.kernel_size, inv_layer.stride
        output_size = np.floor((random_dim - kernel_size + 2 * (kernel_size - 1) // 2) / stride + 1)
        self.assertEqual(out.size()[-1], output_size)

    def test_divide(self):
        random_ch = randint(1, 32)
        random_dim = randint(3, 2048)
        random_stride = randint(1, 5)
        sample = torch.zeros(1, random_ch, random_dim)
        inv_layer = Inv1d(random_ch, num_groups=2, kernel_size=3, stride=random_stride, reduction_ratio=3)
        if random_dim % random_stride:
            with self.assertRaises(AssertionError):
                inv_layer(sample)
        else:
            inv_layer(sample)

    def test_num_groups(self):
        random_ch = randint(1, 32)
        sample = torch.zeros(1, random_ch, 2048)
        mismatch_group = randint(1, 100)
        if random_ch % mismatch_group == 0:
            mismatch_group += 1

        with self.assertRaises(AssertionError):
            inv_layer = Inv1d(random_ch, num_groups=mismatch_group, kernel_size=3, stride=3, reduction_ratio=3)
            inv_layer(sample)


if __name__ == "__main__":
    unittest.main()
