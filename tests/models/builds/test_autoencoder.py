import unittest
from typing import Type, Union, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from zae_engine.models import AutoEncoder, VAE
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


class TestVAE(unittest.TestCase):
    """Unit tests for the VAE class."""

    def setUp(self):
        """Set up common test data and VAE instance."""
        # VAE parameters
        self.block = UNetBlock  # or other block type
        self.ch_in = 3
        self.ch_out = 3
        self.width = 8
        self.layers = [2, 2, 2, 2]
        self.groups = 1
        self.dilation = 1
        self.norm_layer = nn.BatchNorm2d
        self.skip_connect = False
        self.latent_dim = 128

        # Test data creation (batch size 4, channels 3, 256x256 image)
        self.batch_size = 4
        self.channels = self.ch_in
        self.height = 128
        self.width_img = 128
        self.test_input = torch.randn(self.batch_size, self.channels, self.height, self.width_img)

        # Encoder output shape (channels, height, width)
        self.encoder_output_shape = [
            self.width * 8,
            self.height // 16,
            self.width_img // 16,
        ]  # Example encoder output shape

        # VAE instance creation
        self.vae = VAE(
            block=self.block,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            encoder_output_shape=self.encoder_output_shape,
            groups=self.groups,
            dilation=self.dilation,
            norm_layer=self.norm_layer,
            skip_connect=self.skip_connect,
            latent_dim=self.latent_dim,
        )

    def test_forward_pass(self):
        """Test that the VAE forward pass returns reconstructed, mu, and logvar."""
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that outputs are tensors
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)

    def test_output_shapes(self):
        """Test that the output shapes of reconstructed, mu, and logvar are correct."""
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that reconstructed output shape matches input shape
        self.assertEqual(reconstructed.shape, self.test_input.shape)

        # Check that mu and logvar have shape (batch_size, latent_dim)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_skip_connections_enabled(self):
        """Test VAE behavior when skip connections are enabled."""
        # Ensure skip connections are enabled
        self.vae.skip_connect = True

        # Perform forward pass
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.vae.feature_vectors), 0)

        # Additional checks can be added here

    def test_skip_connections_disabled(self):
        """Test VAE behavior when skip connections are disabled."""
        # Disable skip connections
        self.vae.skip_connect = False

        # Perform forward pass
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.vae.feature_vectors), 0)

        # Additional checks can be added here

    def test_reparameterize(self):
        """Test the reparameterization trick."""
        mu = torch.zeros(self.batch_size, self.latent_dim)
        logvar = torch.zeros(self.batch_size, self.latent_dim)

        z = self.vae.reparameterize(mu, logvar)

        # Check that z has shape (batch_size, latent_dim)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # # Since mu=0 and logvar=0, z should be standard normal
        # # Check mean close to 0 and variance close to 1
        # self.assertTrue(torch.allclose(z.mean(dim=0), mu.mean(dim=0), atol=1e-1))
        # self.assertTrue(torch.allclose(z.var(dim=0, unbiased=False), torch.ones(self.latent_dim), atol=1e-1))

    def test_invalid_input_shape(self):
        """Test that VAE raises an error for invalid input shapes."""
        # Invalid input shape (e.g., 3D tensor instead of 4D)
        invalid_input = torch.randn(
            self.batch_size, self.channels, self.height
        )  # Shape: (batch_size, channels, height)

        with self.assertRaises(RuntimeError):
            self.vae(invalid_input)

        # Alternatively, provide input with different spatial dimensions that mismatch encoder_output_shape
        # For example, if encoder_output_shape expects [16, 64, 64], but input size produces [16, 32, 32]
        # Assuming input size [3, 128, 128] would produce encoder_output_shape [16, 32, 32]
        invalid_input_size = torch.randn(self.batch_size, self.channels, 32, 32)  # Different spatial size

        with self.assertRaises(RuntimeError):
            self.vae(invalid_input_size)

    def test_latent_dim(self):
        """Test that changing latent_dim affects mu and logvar dimensions."""
        # Change latent_dim
        new_latent_dim = 256
        self.vae.latent_dim = new_latent_dim

        # Re-initialize fc_mu, fc_logvar, and fc_z with new latent_dim
        self.vae.fc_mu = nn.Linear(self.vae.encoder_output_features, new_latent_dim)
        self.vae.fc_logvar = nn.Linear(self.vae.encoder_output_features, new_latent_dim)
        self.vae.fc_z = nn.Linear(new_latent_dim, self.vae.encoder_output_features)

        # Perform forward pass
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that mu and logvar have shape (batch_size, new_latent_dim)
        self.assertEqual(mu.shape, (self.batch_size, new_latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, new_latent_dim))

    def test_generated_output(self):
        """Test that generated output from random latent vectors has the correct shape."""
        # Generate random latent vectors
        z = torch.randn(self.batch_size, self.vae.latent_dim)

        # Map z back to feature space
        z_mapped = self.vae.fc_z(z)

        # Reshape z_mapped to match encoder_output_shape
        z_reshaped = z_mapped.view(*([self.batch_size] + self.encoder_output_shape))

        # Pass through bottleneck
        feat = self.vae.bottleneck(z_reshaped)

        # Decoder forward pass
        # Note: feature_vectors are cleared at start, so no skip connections
        self.vae.feature_vectors = []  # Ensure feature_vectors are empty

        for up_pool, dec in zip(self.vae.up_pools, self.vae.decoder):
            feat = up_pool(feat)
            if self.vae.skip_connect and len(self.vae.feature_vectors) > 0:
                feat = torch.cat((feat, self.vae.feature_vectors.pop()), dim=1)
            feat = dec(feat)

        # Final output
        generated = self.vae.sig(self.vae.fc(feat))

        # Check that generated output has same shape as test_input
        self.assertEqual(generated.shape, self.test_input.shape)


if __name__ == "__main__":
    unittest.main()
