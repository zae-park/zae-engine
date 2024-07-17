from random import randint, choice

import unittest
import torch

from zae_engine.nn_night.blocks.spatial_attention import SE1d, CBAM1d


class TestSEBlock(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_build(self):
        ch_in = randint(1, 64)
        reduction = randint(1, 8)

        if ch_in % reduction:
            with self.assertRaises(AssertionError):
                model = SE1d(ch_in, reduction)
        else:
            model = SE1d(ch_in, reduction)

    def test_forward(self):
        random_ch = randint(1, 16)
        random_dim = randint(3, 2048)
        sample = torch.rand(size=(1, random_ch, random_dim), dtype=torch.float32)

        model = SE1d(ch_in=random_ch, reduction=1)
        out = model(sample)

        self.assertEqual(sample.shape, out.shape)


class TestCBAMBlock(unittest.TestCase):

    def test_build(self):
        ch_in = randint(1, 64)
        reduction = randint(1, 8)
        kernel_size = 1 + 2 * randint(1, 5)  # guarantee kernel_size is odd number
        conv_pool = conv_pool = choice([True, False])
        if ch_in % reduction:
            with self.assertRaises(AssertionError):
                model = CBAM1d(ch_in=ch_in, reduction=reduction, kernel_size=kernel_size, conv_pool=conv_pool)
        else:
            model = CBAM1d(ch_in=ch_in, reduction=reduction, kernel_size=kernel_size, conv_pool=conv_pool)

    def test_forward(self):
        random_ch = randint(1, 16)
        random_dim = randint(3, 2048)
        random_kernel = 1 + 2 * randint(1, 5)  # guarantee kernel_size is odd number
        sample = torch.rand(size=(1, random_ch, random_dim), dtype=torch.float32)

        model = CBAM1d(ch_in=random_ch, reduction=1, kernel_size=random_kernel, conv_pool=choice([True, False]))

        if random_dim < random_kernel:
            with self.assertRaises(RuntimeError):
                out = model(sample)
        else:
            out = model(sample)
            self.assertEqual(sample.shape, out.shape)


if __name__ == "__main__":
    unittest.main()
