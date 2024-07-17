from random import randint, choice

import unittest
import torch

from zae_engine.nn_night.blocks.resblock import BasicBlock, Bottleneck


class TestBasicBlock(unittest.TestCase):

    def setUp(self) -> None:
        self.ch_in = 16
        self.ch_out = 32
        self.stride = 1
        self.group = 1
        self.dilation = 1

    def test_ch(self):
        ch_in = randint(1, 64)
        ch_out = randint(1, 64)
        model = BasicBlock(ch_in, ch_out, stride=self.stride, groups=self.group, dilation=self.dilation)

    def test_stride(self):
        stride = randint(1, 5)
        model = BasicBlock(self.ch_in, self.ch_out, stride=stride, groups=self.group, dilation=self.dilation)

    def test_group(self):
        group = randint(1, 4)
        if (self.ch_in % group) or (self.ch_out % group):
            with self.assertRaises(AssertionError):
                model = BasicBlock(self.ch_in, self.ch_out, stride=self.stride, groups=group, dilation=self.dilation)
        else:
            model = BasicBlock(self.ch_in, self.ch_out, stride=self.stride, groups=group, dilation=self.dilation)

    def test_dilation(self):
        dilation = randint(2, 5)
        with self.assertRaises(NotImplementedError):
            model = BasicBlock(self.ch_in, self.ch_out, stride=self.stride, groups=self.group, dilation=dilation)

    def test_forward(self):

        random_ch = randint(2, 64)  # BatchNorm: Expected more than 1 value per channel
        random_dim = randint(1, 2048)
        sample = torch.rand(size=(1, random_ch, random_dim, random_dim), dtype=torch.float32)

        model = BasicBlock(random_ch, random_ch, stride=self.stride, groups=self.group, dilation=self.dilation)
        if random_dim > 5:
            out = model(sample)
        else:
            with self.assertRaises(RuntimeError):
                out = model(sample)


class TestBottleneckBlock(unittest.TestCase):

    def setUp(self) -> None:
        self.ch_in = 16
        self.ch_out = 32
        self.stride = 1
        self.group = 1
        self.dilation = 1

    def test_ch(self):
        ch_in = randint(1, 64)
        ch_out = randint(1, 64)
        model = Bottleneck(ch_in, ch_out, stride=self.stride, groups=self.group, dilation=self.dilation)

    def test_stride(self):
        stride = randint(1, 5)
        model = Bottleneck(self.ch_in, self.ch_out, stride=stride, groups=self.group, dilation=self.dilation)

    def test_group(self):
        random_ch = randint(1, 64)
        group = randint(1, 4)
        if random_ch % group:
            with self.assertRaises(AssertionError):
                model = Bottleneck(random_ch, random_ch, stride=self.stride, groups=group, dilation=self.dilation)
        else:
            model = Bottleneck(random_ch, random_ch, stride=self.stride, groups=group, dilation=self.dilation)

    def test_dilation(self):
        dilation = randint(2, 5)

        # with self.assertRaises(AssertionError):
        model = Bottleneck(self.ch_in, self.ch_out, stride=2, groups=self.group, dilation=dilation)
        model = Bottleneck(self.ch_in, self.ch_out, stride=1, groups=self.group, dilation=dilation)

    def test_forward(self):
        random_ch = randint(1, 64)
        random_dim = randint(1, 2048)
        sample = torch.rand(size=(1, random_ch, random_dim, random_dim), dtype=torch.float32)

        model = Bottleneck(random_ch, random_ch, stride=self.stride, groups=self.group, dilation=self.dilation)
        if random_dim > 5:
            out = model(sample)
        else:
            with self.assertRaises(RuntimeError):
                out = model(sample)


if __name__ == "__main__":
    unittest.main()
