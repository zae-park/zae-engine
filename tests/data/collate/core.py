import unittest
from collections import OrderedDict


from zae_engine.data import CollateBase


class TestCollateBase(unittest.TestCase):

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
        output_batch = collator(input_batch)
        self.assertEqual(output_batch["a"], [2, 4, 6])
        self.assertEqual(output_batch["b"], [2, 3, 4])

    def test_io_check_empty_sample_data(self):
        collator = CollateBase(self.fn1)
        sample_data = {}
        with self.assertRaises(ValueError):
            collator.io_check(sample_data)

    def test_add_fn(self):
        collator = CollateBase(self.fn1)
        self.assertEqual(len(collator), 1)
        collator.add_fn("fn2", self.fn2)
        self.assertEqual(len(collator), 2)
        self.assertTrue("fn2" in collator._fn)


if __name__ == "__main__":
    unittest.main()
