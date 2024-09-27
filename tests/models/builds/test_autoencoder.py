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
    """Unit tests for the VAE class, including Conditional VAE (CVAE) functionality."""

    def setUp(self):
        """Set up common test data and VAE/CVAE instances."""
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

        # Test data creation (batch size 4, channels 3, 128x128 image)
        self.batch_size = 4
        self.channels = self.ch_in
        self.height = 128
        self.width_img = 128
        self.test_input = torch.randn(self.batch_size, self.channels, self.height, self.width_img)
        
        # CVAE parameters
        self.condition_dim = 10  # Example condition dimension

        # Condition data creation for CVAE (batch size 4, condition_dim)
        self.condition = torch.randn(self.batch_size, self.condition_dim)

        # Encoder output shape (channels, height, width)
        self.encoder_output_shape = [self.width * 8, self.height // 16, self.width_img // 16]  # Example encoder output shape

        # VAE instance creation (standard VAE)
        self.vae = VAE(
            block=self.block,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            encoder_output_shape=self.encoder_output_shape,
            condition_dim=None,  # Standard VAE
            groups=self.groups,
            dilation=self.dilation,
            norm_layer=self.norm_layer,
            skip_connect=self.skip_connect,
            latent_dim=self.latent_dim
        )
        
        # CVAE instance creation
        self.cvae = VAE(
            block=self.block,
            ch_in=self.ch_in,
            ch_out=self.ch_out,
            width=self.width,
            layers=self.layers,
            encoder_output_shape=self.encoder_output_shape,
            condition_dim=self.condition_dim,  # CVAE
            groups=self.groups,
            dilation=self.dilation,
            norm_layer=self.norm_layer,
            skip_connect=self.skip_connect,
            latent_dim=self.latent_dim
        )

        
    # --- VAE Tests ---

    def test_forward_pass_vae(self):
        """Test that the VAE forward pass returns reconstructed, mu, and logvar."""
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that outputs are tensors
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)

    def test_output_shapes_vae(self):
        """Test that the output shapes of reconstructed, mu, and logvar are correct for VAE."""
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that reconstructed output shape matches input shape
        self.assertEqual(reconstructed.shape, self.test_input.shape)

        # Check that mu and logvar have shape (batch_size, latent_dim)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_skip_connections_enabled_vae(self):
        """Test VAE behavior when skip connections are enabled."""
        # Enable skip connections
        self.vae.skip_connect = True

        # Perform forward pass
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.vae.feature_vectors), 0)

    def test_skip_connections_disabled_vae(self):
        """Test VAE behavior when skip connections are disabled."""
        # Disable skip connections
        self.vae.skip_connect = False

        # Perform forward pass
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.vae.feature_vectors), 0)

    def test_reparameterize_vae(self):
        """Test the reparameterization trick in VAE."""
        mu = torch.zeros(self.batch_size, self.latent_dim)
        logvar = torch.zeros(self.batch_size, self.latent_dim)

        z = self.vae.reparameterize(mu, logvar)

        # Check that z has shape (batch_size, latent_dim)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # # Since mu=0 and logvar=0, z should be standard normal
        # # Check mean close to 0 and variance close to 1
        # self.assertTrue(torch.allclose(z.mean(dim=0), mu.mean(dim=0), atol=1e-1))
        # self.assertTrue(torch.allclose(z.var(dim=0, unbiased=False), torch.ones(self.latent_dim), atol=1e-1))

    def test_invalid_input_shape_vae(self):
        """Test that VAE raises an error for invalid input shapes."""
        # Invalid input shape (e.g., 3D tensor instead of 4D)
        invalid_input = torch.randn(self.batch_size, self.channels, self.height)  # Shape: (batch_size, channels, height)

        with self.assertRaises(RuntimeError):
            self.vae(invalid_input)

        # Alternatively, provide input with different spatial dimensions that mismatch encoder_output_shape
        # For example, if encoder_output_shape expects [16, 8, 8], but input size produces [16, 4, 4]
        invalid_input_size = torch.randn(self.batch_size, self.channels, 32, 32)  # Different spatial size

        with self.assertRaises(RuntimeError):
            self.vae(invalid_input_size)

    def test_latent_dim_vae(self):
        """Test that changing latent_dim affects mu and logvar dimensions in VAE."""
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

    def test_reconstruction_quality_vae(self):
        """Test that the reconstruction loss decreases after a training step for VAE."""
        # Define loss function
        def vae_loss(reconstructed: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
            recon_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_loss

        # Set the model to training mode
        self.vae.train()

        # Define optimizer
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        # Initial loss calculation
        reconstructed, mu, logvar = self.vae(self.test_input)
        initial_loss = vae_loss(reconstructed, self.test_input, mu, logvar).item()

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss = vae_loss(reconstructed, self.test_input, mu, logvar)
        loss.backward()
        optimizer.step()

        # Updated loss calculation
        reconstructed, mu, logvar = self.vae(self.test_input)
        updated_loss = vae_loss(reconstructed, self.test_input, mu, logvar).item()

        # Check that loss has decreased
        self.assertLess(updated_loss, initial_loss)

    def test_generated_output_vae(self):
        """Test that generated output from random latent vectors has the correct shape for VAE."""
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

    # --- CVAE Tests ---

    def test_forward_pass_cvae(self):
        """Test that the CVAE forward pass returns reconstructed, mu, and logvar."""
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that outputs are tensors
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)

    def test_output_shapes_cvae(self):
        """Test that the output shapes of reconstructed, mu, and logvar are correct for CVAE."""
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that reconstructed output shape matches input shape
        self.assertEqual(reconstructed.shape, self.test_input.shape)

        # Check that mu and logvar have shape (batch_size, latent_dim)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_skip_connections_enabled_cvae(self):
        """Test CVAE behavior when skip connections are enabled."""
        # Enable skip connections
        self.cvae.skip_connect = True

        # Perform forward pass
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.cvae.feature_vectors), 0)

    def test_skip_connections_disabled_cvae(self):
        """Test CVAE behavior when skip connections are disabled."""
        # Disable skip connections
        self.cvae.skip_connect = False

        # Perform forward pass
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that feature_vectors are cleared after forward pass
        self.assertEqual(len(self.cvae.feature_vectors), 0)

    def test_reparameterize_cvae(self):
        """Test the reparameterization trick in CVAE."""
        mu = torch.zeros(self.batch_size, self.latent_dim)
        logvar = torch.zeros(self.batch_size, self.latent_dim)

        z = self.cvae.reparameterize(mu, logvar)

        # Check that z has shape (batch_size, latent_dim)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # # Since mu=0 and logvar=0, z should be standard normal
        # # Check mean close to 0 and variance close to 1
        # self.assertTrue(torch.allclose(z.mean(dim=0), mu.mean(dim=0), atol=1e-1))
        # self.assertTrue(torch.allclose(z.var(dim=0, unbiased=False), torch.ones(self.latent_dim), atol=1e-1))

    def test_invalid_input_shape_cvae(self):
        """Test that CVAE raises an error for invalid input shapes or conditions."""
        # Invalid input shape (e.g., 3D tensor instead of 4D)
        invalid_input = torch.randn(self.batch_size, self.channels, self.height)  # Shape: (batch_size, channels, height)

        with self.assertRaises(RuntimeError):
            self.cvae(invalid_input, self.condition)

        # Condition tensor with incorrect dimension
        incorrect_condition = torch.randn(self.batch_size, self.condition_dim + 1)  # Incorrect condition_dim

        with self.assertRaises(ValueError):
            self.cvae(self.test_input, incorrect_condition)

        # Missing condition tensor
        with self.assertRaises(ValueError):
            self.cvae(self.test_input, None)

    def test_latent_dim_cvae(self):
        """Test that changing latent_dim affects mu and logvar dimensions in CVAE."""
        # Change latent_dim
        new_latent_dim = 256
        self.cvae.latent_dim = new_latent_dim

        # Re-initialize fc_mu, fc_logvar, and fc_z with new latent_dim
        self.cvae.fc_mu = nn.Linear(self.cvae.encoder_output_features + self.cvae.condition_dim, new_latent_dim)
        self.cvae.fc_logvar = nn.Linear(self.cvae.encoder_output_features + self.cvae.condition_dim, new_latent_dim)
        self.cvae.fc_z = nn.Linear(new_latent_dim + self.cvae.condition_dim, self.cvae.encoder_output_features)

        # Perform forward pass
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that mu and logvar have shape (batch_size, new_latent_dim)
        self.assertEqual(mu.shape, (self.batch_size, new_latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, new_latent_dim))

    def test_reconstruction_quality_cvae(self):
        """Test that the reconstruction loss decreases after a training step for CVAE."""
        # Define loss function
        def cvae_loss(reconstructed: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
            recon_loss = nn.functional.binary_cross_entropy(reconstructed, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_loss

        # Set the model to training mode
        self.cvae.train()

        # Define optimizer
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=1e-3)

        # Initial loss calculation
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)
        initial_loss = cvae_loss(reconstructed, self.test_input, mu, logvar).item()

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss = cvae_loss(reconstructed, self.test_input, mu, logvar)
        loss.backward()
        optimizer.step()

        # Updated loss calculation
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)
        updated_loss = cvae_loss(reconstructed, self.test_input, mu, logvar).item()

        # Check that loss has decreased
        self.assertLess(updated_loss, initial_loss)

    def test_generated_output_cvae(self):
        """Test that generated output from random latent vectors has the correct shape for CVAE."""
        # Generate random latent vectors
        z = torch.randn(self.batch_size, self.cvae.latent_dim)

        # Concatenate condition with z
        z_cond = torch.cat([z, self.condition], dim=1)

        # Map z_cond back to feature space
        z_mapped = self.cvae.fc_z(z_cond)

        # Reshape z_mapped to match encoder_output_shape
        z_reshaped = z_mapped.view(*([self.batch_size] + self.cvae.encoder_output_shape))

        # Pass through bottleneck
        feat = self.cvae.bottleneck(z_reshaped)

        # Decoder forward pass
        # Ensure feature_vectors are empty
        self.cvae.feature_vectors = []

        for up_pool, dec in zip(self.cvae.up_pools, self.cvae.decoder):
            feat = up_pool(feat)
            if self.cvae.skip_connect and len(self.cvae.feature_vectors) > 0:
                feat = torch.cat((feat, self.cvae.feature_vectors.pop()), dim=1)
            feat = dec(feat)

        # Final output
        generated = self.cvae.sig(self.cvae.fc(feat))

        # Check that generated output has same shape as test_input
        self.assertEqual(generated.shape, self.test_input.shape)

    def test_missing_condition_cvae(self):
        """Test that CVAE raises an error when condition is missing."""
        with self.assertRaises(ValueError):
            self.cvae(self.test_input, None)

    def test_incorrect_condition_dim_cvae(self):
        """Test that CVAE raises an error when condition dimension is incorrect."""
        # Condition with incorrect dimension
        incorrect_condition = torch.randn(self.batch_size, self.condition_dim + 1)

        with self.assertRaises(ValueError):
            self.cvae(self.test_input, incorrect_condition)

    def test_generated_output_shape_cvae(self):
        """Test that generated output from CVAE has the correct shape."""
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that reconstructed output shape matches input shape
        self.assertEqual(reconstructed.shape, self.test_input.shape)

    def test_forward_pass_vae_no_condition(self):
        """Test that the standard VAE operates correctly without condition."""
        reconstructed, mu, logvar = self.vae(self.test_input)

        # Check that outputs are tensors
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)

    def test_forward_pass_cvae_with_condition(self):
        """Test that the CVAE operates correctly with condition."""
        reconstructed, mu, logvar = self.cvae(self.test_input, self.condition)

        # Check that outputs are tensors
        self.assertIsInstance(reconstructed, Tensor)
        self.assertIsInstance(mu, Tensor)
        self.assertIsInstance(logvar, Tensor)

if __name__ == '__main__':
    unittest.main()