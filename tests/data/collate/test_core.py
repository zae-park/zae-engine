import unittest
from collections import OrderedDict
from typing import Iterator


import torch
import numpy as np

from zae_engine.data import CollateBase


class TestCollateBase2(unittest.TestCase):

    def setUp(self):
        # Define some example functions for testing
        def fn1(batch):
            batch["a"] = [x * 2 for x in batch["a"]]
            return batch

        def fn2(batch):
            batch["b"] = [x + 1 for x in batch["b"]]
            return batch

        self.fn1 = fn1
        self.fn2 = fn2

    def test_initialization_with_list_of_functions(self):
        collator = CollateBase(self.fn1, self.fn2)
        self.assertEqual(len(collator), 2)
        self.assertTrue("0" in collator._fn)
        self.assertTrue("1" in collator._fn)

    def test_initialization_with_ordered_dict(self):
        functions = OrderedDict([("fn1", self.fn1), ("fn2", self.fn2)])
        collator = CollateBase(functions)
        self.assertEqual(len(collator), 2)
        self.assertTrue("fn1" in collator._fn)
        self.assertTrue("fn2" in collator._fn)

    def test_io_check_valid_function(self):
        collator = CollateBase(self.fn1)
        sample_data = {"a": [1, 2, 3]}
        collator.io_check(sample_data)
        self.assertTrue(True)  # If no assertion error, test passes

    def test_io_check_invalid_function(self):
        def invalid_fn(batch):
            return {"b": [x for x in batch["a"]]}

        collator = CollateBase(invalid_fn)
        sample_data = {"a": [1, 2, 3]}
        with self.assertRaises(AssertionError):
            collator.io_check(sample_data)

    def test_set_batch_and_io_check(self):
        collator = CollateBase(self.fn1)
        sample_data = {"a": [1, 2, 3]}
        collator.set_batch(sample_data)
        collator.io_check(collator.sample_batch)
        self.assertTrue(True)  # If no assertion error, test passes

    def test_call_method(self):
        collator = CollateBase(self.fn1, self.fn2)
        input_batch = {"a": [1, 2, 3], "b": [1, 2, 3]}
        output_batch = collator([input_batch])
        self.assertEqual(output_batch["a"], [2, 4, 6])
        self.assertEqual(output_batch["b"], [2, 3, 4])

    def test_io_check_empty_sample_data(self):
        collator = CollateBase(self.fn1)
        sample_data = {}
        with self.assertRaises(ValueError):
            collator.io_check(sample_data)

    def test_add_fn(self):
        collator = CollateBase()
        self.assertEqual(len(collator), 0)
        collator.add_fn("fn1", self.fn1)
        self.assertEqual(len(collator), 1)
        collator.add_fn("fn2", self.fn2)
        self.assertEqual(len(collator), 2)
        self.assertTrue("fn2" in collator._fn)


class TestCollateBase(unittest.TestCase):

    def setUp(self):
        self.sample_data = {"x": np.array([1, 2, 3]), "y": np.array([1]), "aux": [0.5], "filename": "sample.txt"}
        self.collate = CollateBase(x_key=["x"], y_key=["y"], aux_key=["aux"])

    def test_len_and_iter(self):
        self.collate.add_fn("dummy", lambda batch: batch)
        self.assertEqual(len(self.collate), 1)
        self.assertIsInstance(iter(self.collate), Iterator)

    def test_set_batch(self):
        self.collate.set_batch(self.sample_data)
        self.assertEqual(self.collate.sample_batch, self.sample_data)

    def test_io_check(self):
        self.collate.set_batch(self.sample_data)
        # Adding dummy function that does nothing
        self.collate.add_fn("dummy", lambda batch: batch)
        self.collate.io_check(self.sample_data)  # Should pass without assertion error

    def test_add_fn(self):
        self.collate.set_batch(self.sample_data)
        self.collate.add_fn("dummy", lambda batch: batch)
        self.assertEqual(len(self.collate), 1)

    def test_accumulate(self):
        batches = [self.sample_data, self.sample_data]
        accumulated = self.collate.accumulate(batches)
        self.assertEqual(accumulated["x"].shape[0], 2)


# Sample preprocessing functions
def preprocess_x(batch):
    batch["x"] = torch.tensor(batch["x"]) * 2
    return batch


def preprocess_y(batch):
    batch["y"] = torch.tensor(batch["y"]) + 1
    return batch


def preprocess_aux(batch):
    batch["aux"] = torch.tensor(batch["aux"]).float()
    return batch


class TestCollateBase(unittest.TestCase):

    def setUp(self):
        """Set up the initial CollateBase instance and sample batch for testing."""
        self.x_key = ["x"]
        self.y_key = ["y"]
        self.aux_key = ["aux"]
        self.collator = CollateBase(x_key=self.x_key, y_key=self.y_key, aux_key=self.aux_key)

        self.sample_batch = {"x": [1, 2, 3], "y": [10], "aux": [0.5], "filename": "sample.txt"}

    def test_add_fn(self):
        """Test if functions are correctly added to the preprocessing flow."""
        self.collator.add_fn("preprocess_x", preprocess_x)
        self.collator.add_fn("preprocess_y", preprocess_y)

        self.assertEqual(len(self.collator), 2)  # Two functions should be added

    def test_set_batch(self):
        """Test setting a sample batch for validation."""
        self.collator.set_batch(self.sample_batch)
        self.assertEqual(self.collator.sample_batch, self.sample_batch)

    def test_io_check(self):
        """Test if io_check validates the structure and content of the sample batch."""
        self.collator.set_batch(self.sample_batch)
        self.collator.add_fn("preprocess_x", preprocess_x)
        self.collator.add_fn("preprocess_y", preprocess_y)

        # Call io_check and ensure no errors occur
        try:
            self.collator.io_check(self.sample_batch)
        except AssertionError:
            self.fail("io_check raised an AssertionError unexpectedly.")

    def test_accumulate(self):
        """Test accumulate function with a list of sample batches."""
        batch_list = [self.sample_batch, self.sample_batch]
        accumulated = self.collator.accumulate(batch_list)

        # Verify that accumulate merges lists properly
        self.assertEqual(len(accumulated["x"]), 2)
        self.assertEqual(len(accumulated["y"]), 2)
        self.assertEqual(len(accumulated["aux"]), 2)

    def test_call_functionality(self):
        """Test the full pipeline, ensuring all functions are applied in sequence."""
        self.collator.add_fn("preprocess_x", preprocess_x)
        self.collator.add_fn("preprocess_y", preprocess_y)
        self.collator.add_fn("preprocess_aux", preprocess_aux)

        batch_list = [self.sample_batch, self.sample_batch]
        processed_batch = self.collator(batch_list)

        # Check the output values
        expected_x = torch.tensor([2, 4, 6])  # Original x * 2
        expected_y = torch.tensor([11])  # Original y + 1
        expected_aux = torch.tensor([0.5], dtype=torch.float32)

        self.assertTrue(torch.equal(processed_batch["x"][0], expected_x))
        self.assertTrue(torch.equal(processed_batch["y"][0], expected_y))
        self.assertTrue(torch.equal(processed_batch["aux"][0], expected_aux))

    def test_partial_io_check(self):
        """Test that io_check only runs for new functions and skips checked ones."""
        self.collator.set_batch(self.sample_batch)
        self.collator.add_fn("preprocess_x", preprocess_x)
        self.collator.add_fn("preprocess_y", preprocess_y)

        # Run io_check for initial functions
        self.collator.io_check(self.sample_batch)

        # Add a new function and check that only it is validated
        self.collator.add_fn("preprocess_aux", preprocess_aux)

        try:
            self.collator.io_check(self.sample_batch)
        except AssertionError:
            self.fail("io_check raised an AssertionError unexpectedly.")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
