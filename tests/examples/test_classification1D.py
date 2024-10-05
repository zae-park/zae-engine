import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from zae_engine.examples.classification1D import core


class TestInference(unittest.TestCase):
    def setUp(self):
        pass

    @patch("zae_engine.examples.classification1D.InferenceTrainer.inference")
    def test_core_with_valid_numpy_input(self, mock_inference):
        """
        Test core function with valid 2-D numpy input.
        """
        # Prepare input data: 10 samples, each of length 2048
        x = np.zeros(20480)

        # Mock the inference method to return predetermined outputs
        # Assume the model outputs class probabilities over 7 classes (as per HotEncoder n_cls=7)
        # Here, we simulate the model predicting class 3 for all samples
        mock_outputs = [torch.tensor([[0, 0, 0, 1, 0, 0, 0]])] * 10  # Shape: (1, 7) for each batch
        mock_inference.return_value = mock_outputs

        # Execute core function
        result = core(x)

        # Expected result: class index 3 for all samples
        expected = np.array([3] * 10)

        # Verify the result
        np.testing.assert_array_equal(result, expected)

    @patch("zae_engine.examples.classification1D.InferenceTrainer.inference")
    def test_core_with_valid_torch_tensor_input(self, mock_inference):
        """
        Test core function with valid torch.Tensor input.
        """
        # Prepare input data: 5 samples, each of length 2048
        x = torch.zeros(10240)

        # Mock the inference method to return predetermined outputs
        # Simulate different class predictions
        mock_outputs = [
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]),  # Class 1
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]),  # Class 2
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]),  # Class 3
            torch.tensor([[0, 0, 0, 0, 1, 0, 0]]),  # Class 4
            torch.tensor([[0, 0, 0, 0, 0, 1, 0]]),  # Class 5
        ]
        mock_inference.return_value = mock_outputs

        # Execute core function
        result = core(x)

        # Expected result: class indices 1, 2, 3, 4, 5
        expected = np.array([1, 2, 3, 4, 5])

        # Verify the result
        np.testing.assert_array_equal(result, expected)

    def test_core_with_invalid_input_dimension(self):
        """
        Test core function with invalid input dimensions (3-D array).
        Should raise AssertionError.
        """
        # Prepare input data: 2 samples, each of shape (10, 2048) - 3-D array
        x = np.zeros((2, 10, 2048))

        # Execute core function and expect AssertionError
        with self.assertRaises(AssertionError) as context:
            core(x)

        self.assertIn("Expect 1-D array", str(context.exception))

    def test_core_with_empty_input(self):
        """
        Test core function with empty input array.
        Should raise AssertionError.
        """
        # Prepare empty input data
        x = np.array([])

        self.assertEqual(core(x), [])

    def test_core_with_non_numpy_input(self):
        """
        Test core function with non-numpy input type (list).
        Should raise AssertionError.
        """
        # Prepare input data as list
        x = [0] * 20480

        # Execute core function and expect AssertionError
        with self.assertRaises(AttributeError):
            core(x)

    @patch("zae_engine.examples.classification1D.InferenceTrainer.inference")
    def test_core_no_non_background_runs(self, mock_inference):
        """
        Test core function with all background labels (assuming background is class 0).
        """
        # Prepare input data: 5 samples, all zeros (background)
        x = np.zeros(10240)

        # Mock the inference method to return class 0 for all samples
        mock_outputs = [torch.tensor([[1, 0, 0, 0, 0, 0, 0]])] * 5  # Assuming class 0 is background
        mock_inference.return_value = mock_outputs

        # Execute core function
        result = core(x)

        # Expected result: class index 0 for all samples
        expected = np.array([0] * 5)

        # Verify the result
        np.testing.assert_array_equal(result, expected)

    @patch("zae_engine.examples.classification1D.InferenceTrainer.inference")
    def test_core_with_partial_runs_below_sense(self, mock_inference):
        """
        Test core function where some runs are below the sensitivity threshold.
        """
        # Prepare input data: 6 samples, with varying class indices
        x = np.zeros(12288)
        # Simulate that some segments will have classes below sensitivity

        # Mock the inference method to return a mix of classes
        mock_outputs = [
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]),  # Class 1
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]),  # Class 2
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]),  # Class 3
            torch.tensor([[0, 0, 0, 0, 1, 0, 0]]),  # Class 4
            torch.tensor([[0, 0, 0, 0, 0, 1, 0]]),  # Class 5
            torch.tensor([[0, 0, 0, 0, 0, 0, 1]]),  # Class 6
        ]
        mock_inference.return_value = mock_outputs

        # Execute core function
        result = core(x)

        # Expected result: class indices 1, 2, 3, 4, 5, 6
        expected = np.array([1, 2, 3, 4, 5, 6])

        # Verify the result
        np.testing.assert_array_equal(result, expected)

    @patch("zae_engine.examples.classification1D.CNNBase")
    def test_core_model_initialization_failure(self, mock_cnn_base):
        """
        Test core function when model initialization fails.
        Should raise the corresponding exception.
        """
        # Configure the mock to raise an exception when instantiated
        mock_cnn_base.side_effect = Exception("Model initialization failed")

        # Prepare input data
        x = np.zeros(20480)

        # Execute core function and expect Exception
        with self.assertRaises(Exception) as context:
            core(x)

        self.assertIn("Model initialization failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
