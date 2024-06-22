import unittest
import torch
from .zae_engine.nn_night import selective_kernel_conv as skconv


class TestSKConv1D(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.ch_in = 3
        self.seq_len = 32

    def test_basic_functionality(self):
        model = skconv.SKConv1D(ch_in=self.ch_in, ch_out=5)
        input_tensor = torch.randn(self.batch_size, self.ch_in, self.seq_len)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5, self.seq_len))

    def test_different_kernel_sizes(self):
        model = skconv.SKConv1D(ch_in=self.ch_in, ch_out=5, kernels=[3, 5, 7])
        input_tensor = torch.randn(self.batch_size, self.ch_in, self.seq_len)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5, self.seq_len))

    def test_output_size(self):
        out_size = 16
        model = skconv.SKConv1D(ch_in=self.ch_in, ch_out=5, out_size=out_size)
        input_tensor = torch.randn(self.batch_size, self.ch_in, self.seq_len)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5, out_size))

    def test_large_input(self):
        large_seq_len = 128
        model = skconv.SKConv1D(ch_in=self.ch_in, ch_out=5)
        input_tensor = torch.randn(self.batch_size, self.ch_in, large_seq_len)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5, large_seq_len))

    def test_identity_out_pool(self):
        model = skconv.SKConv1D(ch_in=self.ch_in, ch_out=5, out_size=None)
        input_tensor = torch.randn(self.batch_size, self.ch_in, self.seq_len)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5, self.seq_len))

    def test_invalid_kernel_size(self):
        with self.assertRaises(AssertionError):
            skconv.SKConv1D(ch_in=self.ch_in, ch_out=5, kernels=[2, 4])  # Even kernel sizes should raise an assertion error.

if __name__ == "__main__":
    unittest.main()
