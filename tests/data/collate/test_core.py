import unittest
import copy
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Any, List, Union
import torch

from zae_engine.data import CollateBase


# Define dummy preprocessing functions for testing
def fn_identity(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Identity function that returns the batch as-is."""
    return batch


def fn_add_key(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Function that adds a new key 'added_key' to the batch."""
    batch["added_key"] = torch.tensor([1.0])
    return batch


def fn_modify_key(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Function that modifies the 'x' key by adding 1 to each element."""
    if "x" in batch and isinstance(batch["x"], torch.Tensor):
        batch["x"] = batch["x"] + 1
    return batch


def fn_remove_key(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Function that removes the 'aux' key from the batch."""
    if "aux" in batch:
        del batch["aux"]
    return batch


class TestCollateBase(unittest.TestCase):
    """Unit tests for the CollateBase class."""

    def setUp(self):
        """Set up common test data and functions."""
        # Sample batch data
        self.sample_batch = {
            "x": torch.tensor([1.0, 2.0, 3.0]),
            "y": torch.tensor([0]),
            "aux": torch.tensor([0.5]),
            "filename": "sample.txt",
        }

        # Batch list
        self.batch_list = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0, 6.0]),
                "y": torch.tensor([1]),
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]

    def test_initialization_with_list_of_functions(self):
        """Test initializing CollateBase with a list of functions."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_key])
        self.assertEqual(len(collator), 2)
        self.assertIn("0", collator._fn)
        self.assertIn("1", collator._fn)
        self.assertEqual(collator._fn["0"], fn_identity)
        self.assertEqual(collator._fn["1"], fn_add_key)

    def test_initialization_with_ordered_dict(self):
        """Test initializing CollateBase with an OrderedDict of functions."""
        functions = OrderedDict([("identity", fn_identity), ("add_key", fn_add_key)])
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=functions)
        self.assertEqual(len(collator), 2)
        self.assertIn("identity", collator._fn)
        self.assertIn("add_key", collator._fn)
        self.assertEqual(collator._fn["identity"], fn_identity)
        self.assertEqual(collator._fn["add_key"], fn_add_key)

    def test_add_fn_method(self):
        """Test adding a new function using add_fn method."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity])
        collator.add_fn("add_key", fn_add_key)
        self.assertEqual(len(collator), 2)
        self.assertIn("add_key", collator._fn)
        self.assertEqual(collator._fn["add_key"], fn_add_key)
        self.assertFalse(collator._fn_checked["add_key"])

    def test_set_batch_method(self):
        """Test setting a sample batch using set_batch method."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_key])
        collator.set_batch(self.sample_batch)
        self.assertEqual(collator.sample_batch, self.sample_batch)
        for key in collator._fn_checked:
            self.assertFalse(collator._fn_checked[key])

    def test_io_check_all_functions(self):
        """Test io_check with check_all=True to validate all functions."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_key])
        collator.set_batch(self.sample_batch)
        collator.io_check(self.sample_batch, check_all=True)
        for key in collator._fn_checked:
            self.assertTrue(collator._fn_checked[key])

    def test_io_check_new_function_only(self):
        """Test io_check with check_all=False to validate only unchecked functions."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_key])
        collator.set_batch(self.sample_batch)
        # Initially, both functions are unchecked
        collator.io_check(self.sample_batch, check_all=False)
        # Both functions should now be checked
        for key in collator._fn_checked:
            self.assertTrue(collator._fn_checked[key])
        # Add a new function
        collator.add_fn("modify_x", fn_modify_key)
        # Since sample_batch is set, add_fn should trigger io_check, setting 'modify_x' to True
        self.assertTrue(collator._fn_checked["modify_x"])  # Changed expectation to True

    def test_io_check_structure_integrity_success(self):
        """Test io_check to ensure structure integrity when functions maintain it."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity])
        collator.set_batch(self.sample_batch)
        # fn_identity does not alter the structure
        collator.io_check(self.sample_batch, check_all=True)
        self.assertTrue(collator._fn_checked["0"])

    def test_io_check_structure_integrity_failure(self):
        """Test io_check to raise AssertionError when functions alter the structure."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_remove_key])
        collator.set_batch(self.sample_batch)
        with self.assertRaises(AssertionError) as context:
            collator.io_check(self.sample_batch, check_all=True)
        self.assertIn("The functions changed the keys of the batch.", str(context.exception))

    def test_call_method_with_functions(self):
        """Test the __call__ method to apply functions in sequence."""

        # Define functions that modify the batch
        def fn_add_extra_key(batch: Dict[str, Any]) -> Dict[str, Any]:
            batch["extra"] = torch.tensor([2.0])
            return batch

        collator = CollateBase(
            x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_extra_key, fn_modify_key]
        )

        # 전달 전에 배치의 복사본을 생성
        batch1 = copy.deepcopy(self.sample_batch)
        batch2 = copy.deepcopy(self.sample_batch)

        processed = collator([batch1, batch2])

        # 나머지 검증 로직은 동일
        self.assertIn("x", processed)
        self.assertIn("y", processed)
        self.assertIn("aux", processed)
        self.assertIn("extra", processed)
        self.assertIn("filename", processed)

        # Check 'x' after modification using allclose
        expected_x = torch.stack([self.sample_batch["x"] + 1, self.sample_batch["x"] + 1], dim=0).unsqueeze(1)
        self.assertTrue(torch.allclose(processed["x"], expected_x))

        # Check 'y'
        expected_y = torch.stack([self.sample_batch["y"], self.sample_batch["y"]], dim=0).squeeze()
        self.assertTrue(torch.allclose(processed["y"], expected_y))

        # Check 'aux' remains unchanged (since fn_identity and fn_add_extra_key do not modify 'aux')
        expected_aux = [self.sample_batch["aux"], self.sample_batch["aux"]]
        self.assertEqual(processed["aux"], expected_aux)

        # Check 'extra' key was added
        expected_extra = [torch.tensor([2.0]), torch.tensor([2.0])]
        self.assertEqual(processed["extra"], expected_extra)

        # Check 'filename' remains unchanged
        expected_filenames = ["sample.txt", "sample.txt"]
        self.assertEqual(processed["filename"], expected_filenames)

    def test_accumulate_method(self):
        """Test the accumulate method to correctly accumulate batches."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        batches = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0, 6.0]),
                "y": torch.tensor([1]),
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]

        accumulated = collator.accumulate(batches)
        self.assertIn("x", accumulated)
        self.assertIn("y", accumulated)
        self.assertIn("aux", accumulated)
        self.assertIn("filename", accumulated)

        # Check 'x' stacking and unsqueeze
        expected_x = torch.stack([batch["x"] for batch in batches], dim=0)
        self.assertTrue(torch.allclose(accumulated["x"], expected_x))

        # Check 'y' stacking and squeeze
        expected_y = torch.stack([batch["y"] for batch in batches], dim=0).squeeze()
        self.assertTrue(torch.allclose(accumulated["y"], expected_y))

        # Check 'aux' stacking and no change since 'aux' is not in x_key or y_key
        expected_aux = [batch["aux"] for batch in batches]
        self.assertEqual(accumulated["aux"], expected_aux)

        # Check 'filename' accumulation
        expected_filenames = ["sample1.txt", "sample2.txt"]
        self.assertEqual(accumulated["filename"], expected_filenames)

    def test_accumulate_with_empty_batches(self):
        """Test the accumulate method with an empty list of batches."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        accumulated = collator.accumulate([])
        self.assertEqual(accumulated, {})

    def test_accumulate_with_type_error(self):
        """Test the accumulate method to raise RuntimeError when stacking fails."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        # 'x' tensors have different shapes to cause stacking error
        batches = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0]),  # Different shape
                "y": torch.tensor([1]),
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]

        with self.assertRaises(RuntimeError):
            collator.accumulate(batches)

    def test_iterator_and_length(self):
        """Test the __iter__ and __len__ methods."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity, fn_add_key])
        self.assertEqual(len(collator), 2)
        functions = list(iter(collator))
        self.assertEqual(functions, [fn_identity, fn_add_key])

    def test_wrap_method(self):
        """Test the wrap method to ensure it correctly wraps the __call__ method."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity])
        wrapped = collator.wrap()
        self.assertTrue(callable(wrapped))
        # Test that wrapped function behaves the same as __call__
        batch = [self.sample_batch]
        processed = wrapped(batch)
        accumulated = collator(batch)
        self.assertEqual(processed.keys(), accumulated.keys())
        for key in processed.keys():
            if isinstance(processed[key], torch.Tensor) and isinstance(accumulated[key], torch.Tensor):
                self.assertTrue(torch.allclose(processed[key], accumulated[key]))
            else:
                self.assertEqual(processed[key], accumulated[key])

    def test_accumulate_non_tensor_keys(self):
        """Test accumulate method with non-tensor keys."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        batches = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0, 6.0]),
                "y": torch.tensor([1]),
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]
        accumulated = collator.accumulate(batches)
        self.assertIn("filename", accumulated)
        self.assertEqual(accumulated["filename"], ["sample1.txt", "sample2.txt"])

    def test_add_fn_with_io_check(self):
        """Test that adding a function after setting a batch triggers io_check."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity])
        collator.set_batch(self.sample_batch)
        collator.add_fn("add_key", fn_add_key)
        self.assertTrue(collator._fn_checked["add_key"])

    def test_add_fn_with_io_check_failure(self):
        """Test that adding a function that alters the structure triggers an error in io_check."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_identity])
        collator.set_batch(self.sample_batch)

        # Define a function that alters the structure by removing a key
        def fn_alter_structure(batch: Dict[str, Any]) -> Dict[str, Any]:
            del batch["filename"]
            return batch

        with self.assertRaises(AssertionError) as context:
            collator.add_fn("remove_filename", fn_alter_structure)
        self.assertIn("The functions changed the keys of the batch.", str(context.exception))

    def test_accumulate_with_missing_key(self):
        """Test accumulate method when some batches are missing keys."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        batches = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0, 6.0]),
                # 'y' key is missing
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]

        # Since 'y' is a required key in y_key, missing 'y' should raise KeyError
        with self.assertRaises(KeyError) as context:
            collator.accumulate(batches)
        self.assertIn("missing required key: 'y'", str(context.exception))

    def test_call_with_no_functions(self):
        """Test calling CollateBase instance with no functions added."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        processed = collator([self.sample_batch, self.sample_batch])
        # Accumulate should simply stack the batches without any modifications
        accumulated = collator.accumulate([self.sample_batch, self.sample_batch])
        # Compare keys
        self.assertEqual(processed.keys(), accumulated.keys())
        # Compare non-tensor keys
        self.assertEqual(processed["filename"], accumulated["filename"])
        # Compare tensor keys using allclose
        for key in ["x", "y"]:
            self.assertTrue(torch.allclose(processed[key], accumulated[key]))

    def test_accumulate_with_empty_key_lists(self):
        """Test accumulate method when x_key and y_key are empty."""
        collator = CollateBase(x_key=[], y_key=[], aux_key=["aux"], functions=[])
        batches = [
            {"aux": torch.tensor([0.5]), "filename": "sample1.txt"},
            {"aux": torch.tensor([1.5]), "filename": "sample2.txt"},
        ]
        accumulated = collator.accumulate(batches)
        self.assertIn("aux", accumulated)
        self.assertIn("filename", accumulated)
        self.assertEqual(accumulated["aux"], [torch.tensor([0.5]), torch.tensor([1.5])])
        self.assertEqual(accumulated["filename"], ["sample1.txt", "sample2.txt"])

    def test_io_check_with_dtype_change(self):
        """Test io_check to ensure dtype remains unchanged when functions maintain structure."""

        # Define a function that modifies 'x' but keeps dtype
        def fn_modify_x_dtype(batch: Dict[str, Any]) -> Dict[str, Any]:
            if "x" in batch:
                batch["x"] = batch["x"] + 1  # dtype remains torch.float
            return batch

        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_modify_x_dtype])
        collator.set_batch(self.sample_batch)
        collator.io_check(self.sample_batch, check_all=True)
        self.assertTrue(collator._fn_checked["0"])

    def test_io_check_with_dtype_change_failure(self):
        """Test io_check to raise AssertionError when dtype is changed."""

        # Define a function that changes dtype of 'x'
        def fn_change_x_dtype(batch: Dict[str, Any]) -> Dict[str, Any]:
            if "x" in batch:
                batch["x"] = batch["x"].double()  # Change dtype to torch.double
            return batch

        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[fn_change_x_dtype])
        collator.set_batch(self.sample_batch)
        with self.assertRaises(AssertionError) as context:
            collator.io_check(self.sample_batch, check_all=True)
        self.assertIn("The dtype of value for key 'x' has changed.", str(context.exception))

    def test_accumulate_with_non_tensor_keys(self):
        """Test accumulate method with non-tensor keys."""
        collator = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"], functions=[])
        batches = [
            {
                "x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([0]),
                "aux": torch.tensor([0.5]),
                "filename": "sample1.txt",
            },
            {
                "x": torch.tensor([4.0, 5.0, 6.0]),
                "y": torch.tensor([1]),
                "aux": torch.tensor([1.5]),
                "filename": "sample2.txt",
            },
        ]
        accumulated = collator.accumulate(batches)
        self.assertIn("filename", accumulated)
        self.assertEqual(accumulated["filename"], ["sample1.txt", "sample2.txt"])


if __name__ == "__main__":
    unittest.main()
