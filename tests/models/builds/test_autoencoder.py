import unittest
import torch

from zae_engine.models import AutoEncoder
from zae_engine.nn_night.blocks import UNetBlock


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        # Define hyperparameters
        self.ch_in = 3
        self.ch_out = 3
        self.width = 64
        self.layers = [2, 2, 2, 2]  # Layer configuration for each stage
        self.image_size = (32, 3, 64, 64)  # (batch_size, channels, height, width)

        # Define model without skip connections
        self.model_no_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=False,
        )

        # Define model with skip connections (U-Net style)
        self.model_with_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=True,
        )

        # Create sample input
        self.input_tensor = torch.randn(self.image_size)

    def test_forward_no_skip(self):
        # Test forward pass without skip connections
        output = self.model_no_skip(self.input_tensor)
        self.assertEqual(
            output.shape, self.image_size, "Output shape should match input shape without skip connections"
        )

    def test_forward_with_skip(self):
        # Test forward pass with skip connections
        output = self.model_with_skip(self.input_tensor)
        self.assertEqual(output.shape, self.image_size, "Output shape should match input shape with skip connections")

    def test_decoder_input_channels_with_skip_connection(self):
        # AutoEncoder with skip connections
        model_skip = AutoEncoder(
            block=UNetBlock,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            skip_connect=True,
        )

        # Forward pass to register hooks and pop feature maps
        input_data = torch.randn(1, self.ch_in, 64, 64)
        model_skip(input_data)

        # Ensure that when skip connections are used, the input channels for the decoder layers are correctly increased
        for up_pool, dec in zip(model_skip.up_pools, model_skip.decoder):
            # Skip connections should concatenate feature maps, doubling the input channels for each decoder stage
            input_channels = dec[0].conv1.in_channels
            self.assertEqual(
                input_channels,
                up_pool.out_channels * 2,
                f"Decoder input channels should be doubled due to skip connections at layer {dec}",
            )


if __name__ == "__main__":
    unittest.main()
