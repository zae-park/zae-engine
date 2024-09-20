# test_shape_check.py
import unittest
import numpy as np
import torch
import pandas as pd

# Import the optimized shape_check decorator
from zae_engine.utils.decorators import shape_check


class TestShapeCheckDecorator(unittest.TestCase):
    """Unit tests for the shape_check decorator."""

    def test_function_with_positional_args_success(self):
        """Test shape_check with a function using positional arguments where shapes match."""

        @shape_check(2)
        def add_arrays(a, b):
            return a + b

        x = np.ones((3, 3))
        y = np.ones((3, 3))

        result = add_arrays(x, y)
        np.testing.assert_array_equal(result, x + y)

    def test_function_with_positional_args_failure(self):
        """Test shape_check with a function using positional arguments where shapes do not match."""

        @shape_check(2)
        def add_arrays(a, b):
            return a + b

        x = np.ones((3, 3))
        y = np.ones((2, 2))

        with self.assertRaises(AssertionError) as context:
            add_arrays(x, y)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_function_with_keyword_args_success(self):
        """Test shape_check with a function using keyword arguments where shapes match."""

        @shape_check("a", "b")
        def add_arrays(a=None, b=None):
            return a + b

        x = np.ones((4, 4))
        y = np.ones((4, 4))

        result = add_arrays(a=x, b=y)
        np.testing.assert_array_equal(result, x + y)

    def test_function_with_keyword_args_failure(self):
        """Test shape_check with a function using keyword arguments where shapes do not match."""

        @shape_check("a", "b")
        def add_arrays(a=None, b=None):
            return a + b

        x = np.ones((4, 4))
        y = np.ones((5, 5))

        with self.assertRaises(AssertionError) as context:
            add_arrays(a=x, b=y)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_method_with_positional_args_success(self):
        """Test shape_check with a method using positional arguments where shapes match."""

        class ArrayProcessor:
            @shape_check(2)
            def process(self, x, y):
                return x + y

        processor = ArrayProcessor()
        x = np.ones((2, 2))
        y = np.ones((2, 2))

        result = processor.process(x, y)
        np.testing.assert_array_equal(result, x + y)

    def test_method_with_positional_args_failure(self):
        """Test shape_check with a method using positional arguments where shapes do not match."""

        class ArrayProcessor:
            @shape_check(2)
            def process(self, x, y):
                return x + y

        processor = ArrayProcessor()
        x = np.ones((2, 2))
        y = np.ones((3, 3))

        with self.assertRaises(AssertionError) as context:
            processor.process(x, y)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_method_with_keyword_args_success(self):
        """Test shape_check with a method using keyword arguments where shapes match."""

        class ArrayProcessor:
            @shape_check("x", "y")
            def process(self, x=None, y=None):
                return x + y

        processor = ArrayProcessor()
        x = np.ones((5, 5))
        y = np.ones((5, 5))

        result = processor.process(x=x, y=y)
        np.testing.assert_array_equal(result, x + y)

    def test_method_with_keyword_args_failure(self):
        """Test shape_check with a method using keyword arguments where shapes do not match."""

        class ArrayProcessor:
            @shape_check("x", "y")
            def process(self, x=None, y=None):
                return x + y

        processor = ArrayProcessor()
        x = np.ones((5, 5))
        y = np.ones((6, 6))

        with self.assertRaises(AssertionError) as context:
            processor.process(x=x, y=y)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_missing_argument(self):
        """Test shape_check when a required argument is missing."""

        @shape_check("a", "b")
        def add_arrays(a, b):
            return a + b

        x = np.ones((3, 3))

        with self.assertRaises(TypeError) as context:
            add_arrays(a=x)
        self.assertIn("missing a required argument", str(context.exception))

    def test_argument_without_shape(self):
        """Test shape_check when an argument does not have a 'shape' attribute."""

        @shape_check("a", "b")
        def add_arrays(a=None, b=None):
            return a + b

        x = np.ones((3, 3))
        y = 5  # Does not have 'shape' attribute

        with self.assertRaises(TypeError) as context:
            add_arrays(a=x, b=y)
        self.assertIn("does not have a 'shape' attribute", str(context.exception))

    def test_insufficient_positional_arguments(self):
        """Test shape_check when not enough positional arguments are provided."""

        @shape_check(3)
        def add_arrays(a, b, c):
            return a + b + c

        x = np.ones((2, 2))
        y = np.ones((2, 2))

        with self.assertRaises(TypeError) as context:
            add_arrays(x, y)  # Missing third argument
        self.assertIn("missing a required argument", str(context.exception))

    def test_no_shapes_to_compare(self):
        """Test shape_check when there are no shapes to compare."""

        with self.assertRaises(ValueError) as context:

            @shape_check(0)
            def dummy_function():
                pass

        self.assertIn("Cannot compare shape of single argument or non-positive number", str(context.exception))

    def test_different_shapes_with_more_dimensions(self):
        """Test shape_check with arrays of different shapes in higher dimensions."""

        @shape_check("a", "b")
        def process_arrays(a=None, b=None):
            return a + b

        x = np.ones((2, 3, 4))
        y = np.ones((2, 3, 5))

        with self.assertRaises(AssertionError) as context:
            process_arrays(a=x, b=y)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_same_shapes_with_more_dimensions(self):
        """Test shape_check with arrays of same shapes in higher dimensions."""

        @shape_check("a", "b", "c")
        def process_arrays(a=None, b=None, c=None):
            return a + b + c

        x = np.ones((2, 3, 4))
        y = np.ones((2, 3, 4))
        z = np.ones((2, 3, 4))

        result = process_arrays(a=x, b=y, c=z)
        np.testing.assert_array_equal(result, x + y + z)

    def test_error_message_contains_function_name(self):
        """Test that error messages contain the function name for easier debugging."""

        @shape_check("a", "b")
        def custom_function(a=None, b=None):
            return a + b

        x = np.ones((2, 2))
        y = np.ones((3, 3))

        with self.assertRaises(AssertionError) as context:
            custom_function(a=x, b=y)
        self.assertIn('An error occurred in function "custom_function"', str(context.exception))

    def test_method_with_default_arguments(self):
        """Test shape_check with a method that has default arguments."""

        class Processor:
            @shape_check("x", "y")
            def compute(self, x=None, y=None):
                return x + y

        processor = Processor()
        x = np.ones((4, 4))
        y = np.ones((4, 4))

        result = processor.compute(x=x, y=y)
        np.testing.assert_array_equal(result, x + y)

    def test_function_with_varargs(self):
        """Test shape_check with a function that uses *args."""

        @shape_check(3)
        def sum_arrays(*args):
            return sum(args)

        x = np.ones((3, 3))
        y = np.ones((3, 3))
        z = np.ones((3, 3))

        result = sum_arrays(x, y, z)
        np.testing.assert_array_equal(result, x + y + z)

    def test_function_with_varargs_shape_mismatch(self):
        """Test shape_check with a function that uses *args and has shape mismatch."""

        @shape_check(3)
        def sum_arrays(*args):
            return sum(args)

        x = np.ones((3, 3))
        y = np.ones((4, 4))
        z = np.ones((3, 3))

        with self.assertRaises(AssertionError) as context:
            sum_arrays(x, y, z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))


if __name__ == "__main__":
    unittest.main()
