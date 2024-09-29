import unittest
from typing import Dict, Any
import numpy as np
import torch

# Import the optimized torch2np decorator
from zae_engine.utils.decorators import torch2np


class TestTorch2NPDecorator(unittest.TestCase):
    """Unit tests for the torch2np decorator."""

    def test_full_conversion_function(self):
        """Test full key conversion on a standalone function."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.float32),
            "y": torch.tensor([0], dtype=torch.float32),
            "aux": [0.5],
            "filename": "sample.txt",
        }

        processed = process_batch(batch)

        # Check 'x'
        self.assertIsInstance(processed["x"], np.ndarray)
        self.assertEqual(processed["x"].dtype, np.float32)
        np.testing.assert_array_equal(processed["x"], np.array([1, 2, 3], dtype=np.float32))

        # Check 'y'
        self.assertIsInstance(processed["y"], np.ndarray)
        self.assertEqual(processed["y"].dtype, np.float32)
        np.testing.assert_array_equal(processed["y"], np.array([0], dtype=np.float32))

        # Ensure other fields are unchanged
        self.assertEqual(processed["aux"], [0.5])
        self.assertEqual(processed["filename"], "sample.txt")

    def test_partial_conversion_function(self):
        """Test partial key conversion on a standalone function."""

        @torch2np(np.int64, "x")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.int64),
            "y": [0],  # Should remain unchanged
            "aux": [0.5],
            "filename": "sample.txt",
        }

        processed = process_batch(batch)

        # Check 'x'
        self.assertIsInstance(processed["x"], np.ndarray)
        self.assertEqual(processed["x"].dtype, np.int64)
        np.testing.assert_array_equal(processed["x"], np.array([1, 2, 3], dtype=np.int64))

        # Ensure other fields are unchanged
        self.assertEqual(processed["y"], [0])
        self.assertEqual(processed["aux"], [0.5])
        self.assertEqual(processed["filename"], "sample.txt")

    def test_n_argument_conversion(self):
        """Test conversion of the first n positional arguments."""

        @torch2np(np.float32, n=3)  # Convert first 3 arguments
        def add_tensors(x, y, z):
            return x + y + z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        result = add_tensors(x, y, z)

        # All three arguments are converted to np.float32, so result should be np.float32
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([12, 15, 18], dtype=np.float32))

    def test_method_conversion(self):
        """Test the decorator on a class method."""

        class Example:
            @torch2np(np.float32, "x", "y")
            def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                return batch

        example = Example()

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.float32),
            "y": torch.tensor([4, 5, 6], dtype=torch.float32),
            "z": torch.tensor([7, 8, 9], dtype=torch.float32),
        }

        processed = example.process(batch)

        # Check 'x'
        self.assertIsInstance(processed["x"], np.ndarray)
        self.assertEqual(processed["x"].dtype, np.float32)
        np.testing.assert_array_equal(processed["x"], np.array([1, 2, 3], dtype=np.float32))

        # Check 'y'
        self.assertIsInstance(processed["y"], np.ndarray)
        self.assertEqual(processed["y"].dtype, np.float32)
        np.testing.assert_array_equal(processed["y"], np.array([4, 5, 6], dtype=np.float32))

        # 'z'는 변환되지 않았으므로 Torch Tensor로 남아있어야 함
        self.assertIsInstance(processed["z"], torch.Tensor)
        self.assertEqual(processed["z"].dtype, torch.float32)
        torch.testing.assert_close(processed["z"], torch.tensor([7, 8, 9], dtype=torch.float32))

    def test_n_argument_exceeds_length(self):
        """Test that specifying n greater than the number of arguments works without error."""

        @torch2np(np.float32, n=5)  # n=5 exceeds the number of arguments (3)
        def add_tensors(x, y, z):
            return x + y + z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        result = add_tensors(x, y, z)

        # All three arguments are converted to np.float32, so result should be np.float32
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([12, 15, 18], dtype=np.float32))

    def test_non_tensor_input(self):
        """Test that non-tensor inputs are not converted."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": [1, 2, 3],  # Not a torch tensor
            "y": [0],  # Not a torch tensor
            "aux": [0.5],
            "filename": "sample.txt",
        }

        processed = process_batch(batch)

        # 'x' and 'y' should remain unchanged as lists
        self.assertEqual(processed["x"], [1, 2, 3])
        self.assertEqual(processed["y"], [0])
        self.assertEqual(processed["aux"], [0.5])
        self.assertEqual(processed["filename"], "sample.txt")

    def test_missing_keys_in_dict(self):
        """Test that missing keys in the dictionary raise KeyError."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.float32),
            "z": torch.tensor([7, 8, 9], dtype=torch.float32),  # 'y' key is missing
        }

        with self.assertRaises(KeyError):
            process_batch(batch)

    def test_invalid_argument_type(self):
        """Test that providing non-dict argument when keys are specified raises TypeError."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Any) -> Any:
            return batch

        # Passing a list instead of a dict
        batch = ["x", "y"]

        with self.assertRaises(TypeError):
            process_batch(batch)

    def test_all_arguments_conversion(self):
        """Test that all torch tensor arguments are converted when no keys or n are specified."""

        @torch2np(np.float32)
        def process_batch(x, y, z):
            return x + y + z

        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        y = torch.tensor([4, 5, 6], dtype=torch.float32)
        z = torch.tensor([7, 8, 9], dtype=torch.float32)

        result = process_batch(x, y, z)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([12, 15, 18], dtype=np.float32))

    def test_conversion_with_some_keys_present(self):
        """Test that only existing keys are converted and others are ignored."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": torch.tensor([1, 2, 3], dtype=torch.float32),
            "z": torch.tensor([7, 8, 9], dtype=torch.float32),  # 'y' key is missing
        }

        # Since 'y' key is missing, KeyError should be raised
        with self.assertRaises(KeyError):
            process_batch(batch)

    def test_multiple_calls(self):
        """Test that the decorator works correctly on multiple calls."""

        @torch2np(np.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch1 = {"x": torch.tensor([1, 2, 3], dtype=torch.float32), "y": torch.tensor([4, 5, 6], dtype=torch.float32)}

        batch2 = {
            "x": torch.tensor([7, 8, 9], dtype=torch.float32),
            "y": torch.tensor([10, 11, 12], dtype=torch.float32),
        }

        processed1 = process_batch(batch1)
        processed2 = process_batch(batch2)

        # Verify batch1
        self.assertIsInstance(processed1["x"], np.ndarray)
        self.assertEqual(processed1["x"].dtype, np.float32)
        np.testing.assert_array_equal(processed1["x"], np.array([1, 2, 3], dtype=np.float32))

        self.assertIsInstance(processed1["y"], np.ndarray)
        self.assertEqual(processed1["y"].dtype, np.float32)
        np.testing.assert_array_equal(processed1["y"], np.array([4, 5, 6], dtype=np.float32))

        # Verify batch2
        self.assertIsInstance(processed2["x"], np.ndarray)
        self.assertEqual(processed2["x"].dtype, np.float32)
        np.testing.assert_array_equal(processed2["x"], np.array([7, 8, 9], dtype=np.float32))

        self.assertIsInstance(processed2["y"], np.ndarray)
        self.assertEqual(processed2["y"].dtype, np.float32)
        np.testing.assert_array_equal(processed2["y"], np.array([10, 11, 12], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
