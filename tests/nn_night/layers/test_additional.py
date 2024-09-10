import unittest
import torch
import torch.nn as nn
from zae_engine.nn_night.layers import Additional


class TestAdditionalLayer(unittest.TestCase):
    def setUp(self):
        # Setup: Define the layers and the input tensors for the test
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)

        self.additional_layer = Additional(self.layer1, self.layer2, self.layer3)

        # Sample input tensors (batch_size=32, input_dim=10)
        self.input1 = torch.randn(32, 10)
        self.input2 = torch.randn(32, 10)
        self.input3 = torch.randn(32, 10)

    def test_forward(self):
        # Test: Ensure that forward pass works with multiple inputs
        inputs = [self.input1, self.input2, self.input3]
        output = self.additional_layer(*inputs)

        # Check the output shape
        self.assertEqual(output.shape, (32, 10))

    def test_shape_mismatch(self):
        # Test: Check if ValueError is raised when output shapes differ
        input_with_different_shape = torch.randn(32, 20)  # Shape mismatch

        with self.assertRaises(ValueError):
            self.additional_layer(self.input1, input_with_different_shape, self.input3)

    def test_input_count_mismatch(self):
        # Test: Check if ValueError is raised when number of inputs doesn't match number of layers
        with self.assertRaises(ValueError):
            self.additional_layer(self.input1, self.input2)  # Missing one input

    def test_output_sum(self):
        # Test: Ensure the output is the sum of the individual layer outputs
        outputs = [
            layer(input) for layer, input in zip(self.additional_layer.layers, [self.input1, self.input2, self.input3])
        ]
        expected_sum = sum(outputs)

        output = self.additional_layer(self.input1, self.input2, self.input3)
        self.assertTrue(torch.allclose(output, expected_sum), "Output does not match the expected sum of layer outputs")


if __name__ == "__main__":
    unittest.main()
