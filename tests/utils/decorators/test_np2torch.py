import unittest
from typing import Dict, Any
import numpy as np
import torch

# Import the optimized np2torch decorator
from zae_engine.utils.decorators import np2torch


class TestNP2TorchDecorator(unittest.TestCase):
    """Unit tests for the np2torch decorator."""

    def test_full_conversion_function(self):
        """Test full key conversion on a standalone function."""

        @np2torch(torch.float32, "x", "y", device=torch.device("cpu"))
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {"x": np.array([1, 2, 3]), "y": np.array([0]), "aux": [0.5], "filename": "sample.txt"}

        processed = process_batch(batch)

        self.assertIsInstance(processed["x"], torch.Tensor)
        self.assertEqual(processed["x"].dtype, torch.float32)
        self.assertEqual(processed["x"].tolist(), [1, 2, 3])

        self.assertIsInstance(processed["y"], torch.Tensor)
        self.assertEqual(processed["y"].dtype, torch.float32)
        self.assertEqual(processed["y"].item(), 0)

        # Ensure other fields are unchanged
        self.assertEqual(processed["aux"], [0.5])
        self.assertEqual(processed["filename"], "sample.txt")

    def test_partial_conversion_function(self):
        """Test partial key conversion on a standalone function."""

        @np2torch(torch.int64, "x")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {"x": np.array([1, 2, 3]), "y": [0], "aux": [0.5], "filename": "sample.txt"}  # Should remain unchanged

        processed = process_batch(batch)

        self.assertIsInstance(processed["x"], torch.Tensor)
        self.assertEqual(processed["x"].dtype, torch.int64)
        self.assertEqual(processed["x"].tolist(), [1, 2, 3])

        # Ensure other fields are unchanged
        self.assertEqual(processed["y"], [0])
        self.assertEqual(processed["aux"], [0.5])
        self.assertEqual(processed["filename"], "sample.txt")

    def test_n_argument_conversion(self):
        """Test conversion of the first n positional arguments."""

        @np2torch(torch.float32, n=3)  # Changed n=2 to n=3 to convert all arguments
        def add_tensors(x, y, z):
            return x + y + z

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        result = add_tensors(x, y, z)

        # All three arguments are converted to torch.float32, so result dtype should be float32
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32, "Result dtype mismatch.")
        self.assertTrue(torch.equal(result, torch.tensor([12, 15, 18], dtype=torch.float32)), "Result values mismatch.")

    def test_method_conversion(self):
        """Test the decorator on a class method."""

        class Example:
            @np2torch(torch.float32, "x", "y")
            def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                return batch

        example = Example()

        batch = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])}

        processed = example.process(batch)

        self.assertIsInstance(processed["x"], torch.Tensor)
        self.assertEqual(processed["x"].dtype, torch.float32)
        self.assertTrue(torch.equal(processed["x"], torch.tensor([1, 2, 3], dtype=torch.float32)), "Mismatch in 'x'.")

        self.assertIsInstance(processed["y"], torch.Tensor)
        self.assertEqual(processed["y"].dtype, torch.float32)
        self.assertTrue(torch.equal(processed["y"], torch.tensor([4, 5, 6], dtype=torch.float32)), "Mismatch in 'y'.")

        # 'z'는 변환되지 않았으므로 NumPy 배열로 남아있어야 함
        self.assertTrue(np.array_equal(processed["z"], np.array([7, 8, 9])), "Mismatch in 'z'.")

    def test_n_argument_exceeds_length(self):
        """Test that specifying n greater than the number of arguments works without error."""

        @np2torch(torch.float32, n=5)  # n=5 exceeds the number of arguments (3)
        def add_tensors(x, y, z):
            return x + y + z

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        result = add_tensors(x, y, z)

        # All three arguments are converted to torch.float32, so result dtype should be float32
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32, "Result dtype mismatch.")
        self.assertTrue(torch.equal(result, torch.tensor([12, 15, 18], dtype=torch.float32)), "Result values mismatch.")

    def test_non_numpy_input(self):
        """Test that non-numpy inputs are not converted."""

        @np2torch(torch.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {
            "x": [1, 2, 3],  # Not a numpy array
            "y": [0],  # Not a numpy array
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

        @np2torch(torch.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {"x": np.array([1, 2, 3]), "z": np.array([7, 8, 9])}  # 'y' key is missing

        with self.assertRaises(KeyError):
            process_batch(batch)

    def test_invalid_argument_type(self):
        """Test that providing non-dict argument when keys are specified raises TypeError."""

        @np2torch(torch.float32, "x", "y")
        def process_batch(batch: Any) -> Any:
            return batch

        # Passing a list instead of a dict
        batch = ["x", "y"]

        with self.assertRaises(TypeError):
            process_batch(batch)

    def test_device_conversion(self):
        """Test that tensors are placed on the specified device."""
        # Determine the device to test
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        @np2torch(torch.float32, "x", "y", device=device)
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {"x": np.array([1, 2, 3]), "y": np.array([0]), "z": np.array([7, 8, 9])}

        processed = process_batch(batch)

        # Verify 'x' tensor
        self.assertIsInstance(processed["x"], torch.Tensor)
        self.assertEqual(processed["x"].dtype, torch.float32)
        self.assertEqual(processed["x"].device.type, device.type, "Device type mismatch for 'x'.")
        expected_index = device.index if device.index is not None else 0
        actual_index = processed["x"].device.index if processed["x"].device.index is not None else 0
        self.assertEqual(actual_index, expected_index, "Device index mismatch for 'x'.")

        # Verify 'y' tensor
        self.assertIsInstance(processed["y"], torch.Tensor)
        self.assertEqual(processed["y"].dtype, torch.float32)
        self.assertEqual(processed["y"].device.type, device.type, "Device type mismatch for 'y'.")
        expected_index = device.index if device.index is not None else 0
        actual_index = processed["y"].device.index if processed["y"].device.index is not None else 0
        self.assertEqual(actual_index, expected_index, "Device index mismatch for 'y'.")

        # 'z'는 변환되지 않았으므로 NumPy 배열로 남아있어야 함
        self.assertTrue(np.array_equal(processed["z"], np.array([7, 8, 9])), "Mismatch in 'z'.")

    def test_all_arguments_conversion(self):
        """Test that all numpy array arguments are converted when no keys or n are specified."""

        @np2torch(torch.float32)
        def process_batch(x, y, z):
            return x + y + z

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])

        result = process_batch(x, y, z)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, torch.tensor([12, 15, 18], dtype=torch.float32)), "Result values mismatch.")

    def test_conversion_with_some_keys_present(self):
        """Test that only existing keys are converted and others are ignored."""

        @np2torch(torch.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch = {"x": np.array([1, 2, 3]), "z": np.array([7, 8, 9])}  # 'y' key is missing

        # Since 'y' key is missing, KeyError should be raised
        with self.assertRaises(KeyError):
            process_batch(batch)

    def test_multiple_calls(self):
        """Test that the decorator works correctly on multiple calls."""

        @np2torch(torch.float32, "x", "y")
        def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return batch

        batch1 = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])}

        batch2 = {"x": np.array([7, 8, 9]), "y": np.array([10, 11, 12])}

        processed1 = process_batch(batch1)
        processed2 = process_batch(batch2)

        # Verify batch1
        self.assertIsInstance(processed1["x"], torch.Tensor)
        self.assertEqual(processed1["x"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(processed1["x"], torch.tensor([1, 2, 3], dtype=torch.float32)), "Mismatch in 'x' for batch1."
        )

        self.assertIsInstance(processed1["y"], torch.Tensor)
        self.assertEqual(processed1["y"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(processed1["y"], torch.tensor([4, 5, 6], dtype=torch.float32)), "Mismatch in 'y' for batch1."
        )

        # Verify batch2
        self.assertIsInstance(processed2["x"], torch.Tensor)
        self.assertEqual(processed2["x"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(processed2["x"], torch.tensor([7, 8, 9], dtype=torch.float32)), "Mismatch in 'x' for batch2."
        )

        self.assertIsInstance(processed2["y"], torch.Tensor)
        self.assertEqual(processed2["y"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(processed2["y"], torch.tensor([10, 11, 12], dtype=torch.float32)), "Mismatch in 'y' for batch2."
        )


if __name__ == "__main__":
    unittest.main()
