import unittest
import torch
import torch.nn as nn
from zae_engine.nn_night.layers import residual_connection


class TestResidualModule(unittest.TestCase):

    def setUp(self):
        # Define a simple sequence of layers
        self.layers = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.BatchNorm2d(3), nn.ReLU())
        self.residual = residual_connection.Residual(*self.layers)

    def test_output_shape(self):
        input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor
        output_tensor = self.residual(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape, "Output shape does not match input shape.")

    def test_residual_connection(self):
        # Define a simple identity layer
        identity_layer = nn.Identity()
        residual = residual_connection.Residual(identity_layer)

        input_tensor = torch.randn(1, 3, 32, 32)
        output_tensor = residual(input_tensor)

        # For an identity layer, the output should be double the input (input + input)
        expected_output = input_tensor + input_tensor
        self.assertTrue(torch.equal(output_tensor, expected_output), "Residual connection did not work as expected.")

    def test_non_identity_layers(self):
        # Test with non-identity layers
        input_tensor = torch.randn(1, 3, 32, 32)
        output_tensor = self.residual(input_tensor)

        # Ensure the output is not just the input tensor (the layers should alter the input)
        self.assertFalse(
            torch.equal(output_tensor, input_tensor),
            "Output is identical to input, layers may not be applied correctly.",
        )

        # Ensure the residual connection is applied
        altered_input = self.layers(input_tensor)
        expected_output = input_tensor + altered_input
        self.assertTrue(
            torch.allclose(output_tensor, expected_output, atol=1e-6),
            "Residual connection did not produce expected output.",
        )


if __name__ == "__main__":
    unittest.main()
