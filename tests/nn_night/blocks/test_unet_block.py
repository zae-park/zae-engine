import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from zae_engine.nn_night.blocks import unet_block


class TestUNetBlock(unittest.TestCase):

    def setUp(self):
        # Create a UNetBlock instance with default settings
        self.unet_block = unet_block.UNetBlock(ch_in=3, ch_out=3)
        self.unet_block_stride = unet_block.UNetBlock(ch_in=3, ch_out=3, stride=2)

    def test_output_shape(self):
        input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
        output_tensor = self.unet_block(input_tensor)
        self.assertEqual(
            input_tensor.shape, output_tensor.shape, "Output shape does not match input shape with stride 1."
        )

        output_tensor_stride = self.unet_block_stride(input_tensor)
        expected_shape = (1, 3, 32, 32)  # Expected shape with stride 2
        self.assertEqual(
            output_tensor_stride.shape, expected_shape, "Output shape does not match expected shape with stride 2."
        )

    def test_forward_pass(self):
        input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
        output_tensor = self.unet_block(input_tensor)

        # Ensure the output is not just the input tensor (the layers should alter the input)
        self.assertFalse(
            torch.equal(output_tensor, input_tensor),
            "Output is identical to input, layers may not be applied correctly.",
        )

    def test_downsampling(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        output_tensor_stride = self.unet_block_stride(input_tensor)

        # Check if downsampling was applied correctly
        self.assertEqual(
            output_tensor_stride.shape[2:], (32, 32), "Downsampling did not reduce the spatial dimensions as expected."
        )

    def test_parameters_update(self):
        input_tensor = torch.randn(1, 3, 64, 64, requires_grad=True)
        output_tensor = self.unet_block(input_tensor)
        output_tensor.sum().backward()

        for param in self.unet_block.parameters():
            if param.grad is not None:
                self.assertIsNotNone(param.grad, "Gradient not computed for parameter.")
                self.assertGreater(torch.sum(param.grad).item(), 0, "Gradient should not be zero.")


if __name__ == "__main__":
    unittest.main()
