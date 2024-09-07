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

    def test_feature_hook(self):
        # Check that feature vectors are captured for skip connections
        _ = self.model_with_skip(self.input_tensor)
        self.assertGreater(
            len(self.model_with_skip.feature_vectors), 0, "Feature vectors should be captured with skip connections"
        )


if __name__ == "__main__":
    unittest.main()
