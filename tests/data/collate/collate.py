import unittest
from collections import OrderedDict
from typing import Callable, Union, OrderedDict as OrderedDictType
import torch

from zae_engine.data import CollateBase


class TestCollateBase(unittest.TestCase):

    def test_initialization_with_list_of_functions(self):
        def fn1(batch):
            batch["a"] = batch["a"] * 2
            return batch

        def fn2(batch):
            batch["b"] = batch["b"] + 1
            return batch

        collator = CollateBase(fn1, fn2)
        self.assertEqual(len(collator), 2)
        self.assertTrue("0" in collator._fn)
        self.assertTrue("1" in collator._fn)

    def test_initialization_with_ordered_dict(self):
        def fn1(batch):
            batch["a"] = batch["a"] * 2
            return batch

        def fn2(batch):
            batch["b"] = batch["b"] + 1
            return batch

        functions = OrderedDict([("fn1", fn1), ("fn2", fn2)])
        collator = CollateBase(functions)
        self.assertEqual(len(collator), 2)
        self.assertTrue("fn1" in collator._fn)
        self.assertTrue("fn2" in collator._fn)

    def test_io_check_valid_function(self):
        def fn(batch):
            batch["a"] = batch["a"] * 2
            return batch

        collator = CollateBase(fn)
        sample_data = {"a": torch.tensor(1)}
        collator.io_check(sample_data)
        self.assertTrue(True)  # If no assertion error, test passes

    def test_io_check_invalid_function(self):
        def fn(batch):
            return {"b": batch["a"]}

        collator = CollateBase(fn)
        sample_data = {"a": torch.tensor(1)}
        with self.assertRaises(AssertionError):
            collator.io_check(sample_data)

    def test_set_batch_and_io_check(self):
        def fn(batch):
            batch["a"] = batch["a"] * 2
            return batch

        collator = CollateBase(fn)
        sample_data = {"a": torch.tensor(1)}
        collator.set_batch(sample_data)
        collator.io_check(collator.sample_batch)
        self.assertTrue(True)  # If no assertion error, test passes

    def test_call_method(self):
        def fn1(batch):
            batch["a"] = batch["a"] * 2
            return batch

        def fn2(batch):
            batch["b"] = batch["b"] + 1
            return batch

        collator = CollateBase(fn1, fn2)
        input_batch = {"a": torch.tensor(1), "b": torch.tensor(1)}
        output_batch = collator(input_batch)
        self.assertEqual(output_batch["a"], torch.tensor(2))
        self.assertEqual(output_batch["b"], torch.tensor(2))

    def test_io_check_empty_sample_data(self):
        def fn(batch):
            batch["a"] = batch["a"] * 2
            return batch

        collator = CollateBase(fn)
        sample_data = {}
        with self.assertRaises(ValueError):
            collator.io_check(sample_data)

    def test_add_fn(self):
        def fn1(batch):
            batch["a"] = batch["a"] * 2
            return batch

        def fn2(batch):
            batch["b"] = batch["b"] + 1
            return batch

        collator = CollateBase(fn1)
        self.assertEqual(len(collator), 1)
        collator.add_fn("fn2", fn2)
        self.assertEqual(len(collator), 2)
        self.assertTrue("fn2" in collator._fn)


if __name__ == "__main__":
    unittest.main()
