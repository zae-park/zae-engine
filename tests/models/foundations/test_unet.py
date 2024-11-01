import unittest
from unittest.mock import patch, MagicMock
import io
import torch
from contextlib import redirect_stdout


from zae_engine.models.builds import autoencoder
from zae_engine.nn_night import blocks
from zae_engine.models.foundations import unet_brain
from zae_engine.models import unet


class TestUNetBrain(unittest.TestCase):
    def test_unet_brain_model_structure(self):
        """Test if the unet_brain function returns an AutoEncoder instance with correct structure."""
        model = unet_brain(pretrained=False)
        self.assertIsInstance(model, autoencoder.AutoEncoder)
        # Additional checks for model structure
        self.assertTrue(hasattr(model, "encoder"))
        self.assertTrue(hasattr(model, "decoder"))

    def test_unet_brain_forward_pass(self):
        """Test a forward pass through the unet_brain model."""
        model = unet_brain(pretrained=False)
        model.eval()
        input_tensor = torch.randn(1, 3, 256, 256)  # Random input tensor
        with torch.no_grad():
            output = model(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1, 256, 256))  # Check output shape

    @patch("torch.hub.load_state_dict_from_url")
    def test_unet_brain_pretrained(self, mock_load_state_dict_from_url):
        """Test if the unet_brain function correctly loads pretrained weights."""
        # Create mock weights
        mock_state_dict = {"encoder.body.0.0.conv1.weight": torch.randn(32, 3, 3, 3)}
        mock_load_state_dict_from_url.return_value = mock_state_dict

        model = unet_brain(pretrained=True)

        # Verify that the mock function was called
        mock_load_state_dict_from_url.assert_called_once()

        # Check if the weights were loaded into the model
        model_state_dict = model.state_dict()
        for key in mock_state_dict.keys():
            self.assertTrue(torch.equal(model_state_dict[key], mock_state_dict[key]))

    def test_brain_weight_mapper_with_unexpected_keys(self):
        """Test if _brain_weight_mapper prints keys that are not mapped."""
        # Create source weights with some keys that won't match destination weights
        src_weight = {
            "unexpected_key1": torch.randn(32, 3, 3, 3),
            "unexpected_key2": torch.randn(64, 32, 3, 3),
            "encoder1.enc1.conv1.weight": torch.randn(32, 3, 3, 3),  # This key should map
        }
        model = unet_brain(pretrained=False)
        dst_weight = model.state_dict()

        # Capture the output printed by _brain_weight_mapper
        f = io.StringIO()
        with redirect_stdout(f):
            mapped_weight = unet._brain_weight_mapper(src_weight, dst_weight.copy())
        output = f.getvalue()

        # Check if unexpected keys were printed
        self.assertIn("unexpected_key1", output)
        self.assertIn("unexpected_key2", output)
        # Confirm that the expected key was not printed (since it should be mapped)
        self.assertNotIn("encoder1.enc1.conv1.weight", output)

    def test_brain_weight_mapper_no_unexpected_keys(self):
        """Test if _brain_weight_mapper does not print when all keys are mapped."""
        # Create source weights where all keys should be mapped
        src_weight = {
            "encoder1.enc1conv1.weight": torch.randn(32, 3, 3, 3),
            "encoder2.enc2conv1.weight": torch.randn(64, 32, 3, 3),
            "bottleneck.conv1.weight": torch.randn(128, 64, 3, 3),
            # Add other keys that are expected to be mapped
        }
        model = unet_brain(pretrained=False)
        dst_weight = model.state_dict()

        # Capture the output printed by _brain_weight_mapper
        f = io.StringIO()
        with redirect_stdout(f):
            mapped_weight = unet._brain_weight_mapper(src_weight, dst_weight.copy())
        output = f.getvalue()

        # Since all keys are mapped, there should be no output
        self.assertEqual(output.strip(), "")

    @patch("torch.hub.load_state_dict_from_url")
    def test_unet_brain_pretrained_forward_pass(self, mock_load_state_dict_from_url):
        """Test a forward pass with the pretrained unet_brain model."""
        # Create mock weights
        mock_state_dict = {"encoder.body.0.0.conv1.weight": torch.randn(32, 3, 3, 3)}
        mock_load_state_dict_from_url.return_value = mock_state_dict

        model = unet_brain(pretrained=True)
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    @patch("torch.hub.load_state_dict_from_url", side_effect=Exception("Download failed"))
    def test_unet_brain_pretrained_download_failure(self, mock_load_state_dict_from_url):
        """Test unet_brain behavior when pretrained weights download fails."""
        with self.assertRaises(Exception) as context:
            unet_brain(pretrained=True)
        self.assertTrue("Download failed" in str(context.exception))

    def test_unet_brain_model_parameters(self):
        """Test if the unet_brain model has the expected number of parameters."""
        model = unet_brain(pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        # Set the expected number of parameters (adjust as needed)
        expected_params = 1000000  # Example value
        self.assertGreater(total_params, expected_params)

    def test_unet_brain_output_values(self):
        """Test if the unet_brain model outputs reasonable values."""
        model = unet_brain(pretrained=False)
        model.eval()
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(input_tensor)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


if __name__ == "__main__":
    unittest.main()
