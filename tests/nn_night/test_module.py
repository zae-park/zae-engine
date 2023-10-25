import unittest

from zae_engine.data_pipeline import example_ecg
from zae_engine.nn_night import *


class TestSEModule(unittest.TestCase):
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
        cls.EX_beat = example_ecg(0)

    @classmethod
    def get_attribute(cls):
        example_10sec, example_beat = cls.EX_10sec, cls.EX_beat
        if example_10sec is None or example_beat is None:
            raise ValueError
        return example_10sec, example_beat

    def setUp(self) -> None:
        self.ex_10sec, self.ex_beat = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_module_structure(self):
        ex_10sec, ex_beat = self.get_attribute()

        with self.assertRaises(AttributeError):
            se = SE(ex_10sec.size()[1], reduction=4, spatial=True, bias=False)
            fc = se.fc

        with self.assertRaises(AttributeError):
            se = SE(ex_10sec.size()[1], reduction=4, spatial=False, bias=False)
            pool = se.ch_pool

    def test_shape(self):
        ex_10sec, ex_beat = self.get_attribute()
        output = SE(ex_10sec.size()[1], reduction=4)(ex_10sec)
        input_batch, input_channel, input_dim = ex_10sec.size()
        output_batch, output_channel, output_dim = output.size()

        self.assertEqual(input_batch, output_batch)
        self.assertEqual(input_channel, output_channel)
        self.assertEqual(input_dim, output_dim)

    def test_reduction(self):
        ex_10sec, ex_beat = self.get_attribute()
        mismatch_reduction = 3
        if ex_10sec.size()[1] % mismatch_reduction == 0:
            mismatch_reduction += 1

        with self.assertRaises(AssertionError):
            se_module = SE(ex_10sec.size()[1], reduction=mismatch_reduction, spatial=False)
            se_module(ex_10sec)


if __name__ == "__main__":
    unittest.main()
