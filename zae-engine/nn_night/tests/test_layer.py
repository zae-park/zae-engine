import random
import unittest

import torch
import numpy as np

from zae-engine.data import load_example
from .._layer import Inv1d


class TestInv1d(unittest.TestCase):
    EX_10sec = None
    EX_beat = None
    random_channel_size = None
    dim = 2500

    @classmethod
    def setUpClass(cls) -> None:
        candidate_channels = torch.linspace(28, 128, 2)
        choice = torch.randint(0, len(candidate_channels), (1,))[0]
        cls.random_channel_size = int(candidate_channels[choice])

        cls.EX_10sec = torch.randn(1, cls.random_channel_size, cls.dim)
        cls.EX_beat = load_example(0)

    @classmethod
    def get_attribute(cls):
        inv_layer = Inv1d(cls.EX_10sec.size()[1], num_groups=2, kernel_size=3, stride=5, reduction_ratio=3)
        example_10sec, example_beat = cls.EX_10sec, cls.EX_beat
        if example_10sec is None or example_beat is None:
            raise ValueError
        return example_10sec, example_beat, inv_layer

    def setUp(self) -> None:
        self.ex_10sec, self.ex_beat, self.inv_layer = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_shape(self):
        ex_10sec, ex_beat, inv_layer = self.get_attribute()
        output = inv_layer(ex_10sec)
        kernel_size, stride = inv_layer.kernel_size, inv_layer.stride
        output_size = np.floor((ex_10sec.size()[-1] - kernel_size + 2 * (kernel_size - 1) // 2) / stride + 1)
        print(output_size)
        self.assertEqual(output.size()[-1], output_size)

    def test_divide(self):
        ex_10sec, ex_beat, _ = self.get_attribute()
        mismatch_stride = 3
        if ex_10sec.size()[-1] % mismatch_stride == 0:
            mismatch_stride += 1

        with self.assertRaises(AssertionError):
            inv_layer = Inv1d(ex_10sec.size()[1], num_groups=2, kernel_size=3, stride=mismatch_stride,
                              reduction_ratio=3)
            inv_layer(ex_10sec)

    def test_num_groups(self):
        ex_10sec, ex_beat, _ = self.get_attribute()
        mismatch_group = random.randint(1, 100)
        if ex_10sec.size()[1] % mismatch_group == 0:
            mismatch_group += 1

        with self.assertRaises(AssertionError):
            inv_layer = Inv1d(ex_10sec.size()[1], mismatch_group, kernel_size=3, stride=3, reduction_ratio=3)
            inv_layer(ex_10sec)


if __name__ == '__main__':
    unittest.main()
