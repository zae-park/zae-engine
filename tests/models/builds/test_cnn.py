import unittest

from random import randint
import torch

from zae_engine.models.builds import cnn
from zae_engine.nn_night.blocks import BasicBlock, Bottleneck


class TestResnet(unittest.TestCase):
    test_sample = None

    @classmethod
    def setUpClass(cls) -> None:
        # cls.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        pass

    def setUp(self) -> None:
        self.ch_in = randint(1, 16)
        self.width = randint(1, 16)
        self.ch_out = randint(1, 1000)
        self.layers = [randint(1, 3)] * randint(1, 5)

        # self.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        self.test_sample = torch.randn((2, 3, 256, 256))

    def tearDown(self) -> None:
        pass

    def test_CNNBase(self):
        model1 = cnn.CNNBase(BasicBlock, ch_in=self.ch_in, ch_out=self.ch_out, width=self.width, layers=self.layers)
        model2 = cnn.CNNBase(Bottleneck, ch_in=self.ch_in, ch_out=self.ch_out, width=self.width, layers=self.layers)
        output1 = model1(self.test_sample)
        output2 = model2(self.test_sample)

        self.assertEqual(output1.size(), torch.Size([2, self.ch_out]))
        self.assertEqual(output2.size(), torch.Size([2, self.ch_out]))


if __name__ == "__main__":
    unittest.main()
