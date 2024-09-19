# test_shape_check.py
import unittest
import numpy as np
import torch
import pandas as pd

# Import the optimized shape_check decorator
from zae_engine.utils.decorators import shape_check


# Define SampleClass outside of TestShapeCheckDecorator
class SampleClass:
    @shape_check(2)
    def add_arrays(self, x, y, z):
        return x + y + z

    @shape_check("x", "y")
    def add_arrays_kwargs(self, **kwargs):
        return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)


class TestShapeCheckDecorator(unittest.TestCase):
    """Unit tests for the shape_check decorator."""

    # -----------------------
    # NumPy Array Tests
    # -----------------------

    def test_shape_check_positional_args_matching_numpy(self):
        """Test that decorator passes when first n positional NumPy arguments have the same shape."""

        @shape_check(2)
        def add_arrays(x, y, z):
            return x + y + z

        x = np.zeros((3, 3))
        y = np.ones((3, 3))
        z = np.full((3, 3), 2)

        result = add_arrays(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_positional_args_not_matching_numpy(self):
        """Test that decorator raises AssertionError when first n positional NumPy arguments have different shapes."""

        @shape_check(2)
        def add_arrays(x, y, z):
            return x + y + z

        x = np.zeros((3, 3))
        y = np.ones((4, 3))  # Different shape
        z = np.full((3, 3), 2)

        with self.assertRaises(AssertionError) as context:
            add_arrays(x, y, z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_shape_check_keyword_args_matching_numpy(self):
        """Test that decorator passes when specified keyword NumPy arguments have the same shape."""

        @shape_check("x", "y")
        def add_arrays(**kwargs):
            return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)

        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_arrays(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_extra_kwargs_not_checked_numpy(self):
        """Test that decorator correctly ignores extra keyword arguments beyond specified keys."""

        @shape_check("x", "y")
        def add_arrays(**kwargs):
            return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)

        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_arrays(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    # -----------------------
    # PyTorch Tensor Tests
    # -----------------------

    def test_shape_check_positional_args_matching_torch(self):
        """Test that decorator passes when first n positional PyTorch tensor arguments have the same shape."""

        @shape_check(2)
        def add_tensors(x, y, z):
            return x + y + z

        x = torch.zeros((3, 3))
        y = torch.ones((3, 3))
        z = torch.full((3, 3), 2)

        result = add_tensors(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)

    def test_shape_check_positional_args_not_matching_torch(self):
        """Test that decorator raises AssertionError when first n positional PyTorch tensor arguments have different shapes."""

        @shape_check(2)
        def add_tensors(x, y, z):
            return x + y + z

        x = torch.zeros((3, 3))
        y = torch.ones((4, 3))  # Different shape
        z = torch.full((3, 3), 2)

        with self.assertRaises(AssertionError) as context:
            add_tensors(x, y, z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_shape_check_keyword_args_matching_torch(self):
        """Test that decorator passes when specified keyword PyTorch tensor arguments have the same shape."""

        @shape_check("x", "y")
        def add_tensors(**kwargs):
            return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)

        x = torch.zeros((2, 2))
        y = torch.ones((2, 2))
        z = torch.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_tensors(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)

    def test_shape_check_extra_kwargs_not_checked_torch(self):
        """Test that decorator correctly ignores extra keyword PyTorch tensor arguments beyond specified keys."""

        @shape_check("x", "y")
        def add_tensors(**kwargs):
            return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)

        x = torch.zeros((2, 2))
        y = torch.ones((2, 2))
        z = torch.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_tensors(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)

    # -----------------------
    # Method Tests
    # -----------------------

    def test_shape_check_method_positional_args_matching_numpy(self):
        """Test that decorator passes when first n positional NumPy arguments of a method have the same shape."""
        sample = SampleClass()
        x = np.zeros((3, 3))
        y = np.ones((3, 3))
        z = np.full((3, 3), 2)  # z is not checked

        result = sample.add_arrays(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_method_positional_args_not_matching_numpy(self):
        """Test that decorator raises AssertionError when first n positional NumPy arguments of a method have different shapes."""
        sample = SampleClass()
        x = np.zeros((3, 3))
        y = np.ones((4, 3))  # Different shape
        z = np.full((3, 3), 2)

        with self.assertRaises(AssertionError) as context:
            sample.add_arrays(x, y, z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_shape_check_method_keyword_args_matching_numpy(self):
        """Test that decorator passes when specified keyword NumPy arguments of a method have the same shape."""
        sample = SampleClass()
        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = sample.add_arrays_kwargs(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_method_keyword_args_not_matching_numpy(self):
        """Test that decorator raises AssertionError when specified keyword NumPy arguments of a method have different shapes."""
        sample = SampleClass()
        x = np.zeros((2, 2))
        y = np.ones((3, 3))  # Different shape
        z = np.full((2, 2), 2)

        with self.assertRaises(AssertionError) as context:
            sample.add_arrays_kwargs(x=x, y=y, z=z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    # -----------------------
    # Error Handling Tests
    # -----------------------

    def test_shape_check_non_shape_argument_in_positional_numpy(self):
        """Test that decorator raises TypeError when positional NumPy arguments do not have 'shape' attribute."""

        @shape_check(2)
        def add_elements(x, y):
            return x + y

        x = np.zeros((3, 3))
        y = [1, 2, 3]  # Does not have 'shape'

        with self.assertRaises(TypeError) as context:
            add_elements(x, y)
        self.assertIn("does not have a 'shape' attribute", str(context.exception))

    def test_shape_check_non_shape_argument_in_keyword_numpy(self):
        """Test that decorator raises TypeError when keyword NumPy arguments do not have 'shape' attribute."""

        @shape_check("x", "y")
        def add_elements(**kwargs):
            return kwargs["x"] + kwargs["y"]

        x = np.zeros((3, 3))
        y = [1, 2, 3]  # Does not have 'shape'

        with self.assertRaises(TypeError) as context:
            add_elements(x=x, y=y)
        self.assertIn("does not have a 'shape' attribute", str(context.exception))

    def test_shape_check_non_shape_argument_in_positional_torch(self):
        """Test that decorator raises TypeError when positional PyTorch tensor arguments do not have 'shape' attribute."""

        @shape_check(2)
        def add_elements(x, y):
            return x + y

        x = torch.zeros((3, 3))
        y = [1, 2, 3]  # Does not have 'shape'

        with self.assertRaises(TypeError) as context:
            add_elements(x, y)
        self.assertIn("does not have a 'shape' attribute", str(context.exception))

    def test_shape_check_non_shape_argument_in_keyword_torch(self):
        """Test that decorator raises TypeError when keyword PyTorch tensor arguments do not have 'shape' attribute."""

        @shape_check("x", "y")
        def add_elements(**kwargs):
            return kwargs["x"] + kwargs["y"]

        x = torch.zeros((3, 3))
        y = [1, 2, 3]  # Does not have 'shape'

        with self.assertRaises(TypeError) as context:
            add_elements(x=x, y=y)
        self.assertIn("does not have a 'shape' attribute", str(context.exception))

    def test_shape_check_invalid_n(self):
        """Test that decorator raises ValueError when n is 1 or less."""
        with self.assertRaises(ValueError) as context:

            @shape_check(1)
            def add_elements(x):
                return x

        self.assertIn("Cannot compare shape of single argument or non-positive number", str(context.exception))

    def test_shape_check_invalid_keys(self):
        """Test that decorator raises ValueError when keys are not all strings."""
        with self.assertRaises(ValueError) as context:

            @shape_check("x", 2)
            def add_elements(**kwargs):
                return kwargs["x"] + kwargs["y"]

        self.assertIn(
            "When multiple arguments are provided, they must be strings representing keyword argument names",
            str(context.exception),
        )

    def test_shape_check_missing_keyword_argument(self):
        """Test that decorator raises KeyError when a specified keyword argument is missing."""

        @shape_check("x", "y")
        def add_elements(**kwargs):
            return kwargs["x"] + kwargs["y"]

        x = np.zeros((3, 3))
        # 'y' is missing

        with self.assertRaises(KeyError) as context:
            add_elements(x=x)
        self.assertIn('Keyword argument "y" not found.', str(context.exception))

    def test_shape_check_missing_positional_argument(self):
        """Test that decorator raises AssertionError when not enough positional arguments are provided."""

        @shape_check(2)
        def add_elements(x, y):
            return x + y

        x = np.zeros((3, 3))
        # y is missing

        with self.assertRaises(AssertionError) as context:
            add_elements(x)
        self.assertIn("Expected at least 2 positional arguments", str(context.exception))

    def test_shape_check_no_shape_to_compare(self):
        """Test that decorator raises AssertionError when no shapes are provided."""

        @shape_check(2)
        def add_elements(x, y):
            return x + y

        # No arguments provided
        with self.assertRaises(AssertionError) as context:
            add_elements()
        self.assertIn("Expected at least 2 positional arguments", str(context.exception))

    def test_shape_check_no_keys_or_n(self):
        """Test that decorator raises ValueError when no keys or n are provided."""
        with self.assertRaises(ValueError) as context:

            @shape_check()
            def add(x, y):
                return x + y

        self.assertIn(
            "At least one key or an integer specifying number of positional arguments must be provided.",
            str(context.exception),
        )

    def test_shape_check_method_positional_args_matching_numpy(self):
        """Test that decorator passes when first n positional NumPy arguments of a method have the same shape."""
        sample = SampleClass()
        x = np.zeros((3, 3))
        y = np.ones((3, 3))
        z = np.full((3, 3), 2)  # z is not checked

        result = sample.add_arrays(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_method_positional_args_not_matching_numpy(self):
        """Test that decorator raises AssertionError when first n positional NumPy arguments of a method have different shapes."""
        sample = SampleClass()
        x = np.zeros((3, 3))
        y = np.ones((4, 3))  # Different shape
        z = np.full((3, 3), 2)

        with self.assertRaises(AssertionError) as context:
            sample.add_arrays(x, y, z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_shape_check_method_keyword_args_matching_numpy(self):
        """Test that decorator passes when specified keyword NumPy arguments of a method have the same shape."""
        sample = SampleClass()
        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = sample.add_arrays_kwargs(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_method_keyword_args_not_matching_numpy(self):
        """Test that decorator raises AssertionError when specified keyword NumPy arguments of a method have different shapes."""
        sample = SampleClass()
        x = np.zeros((2, 2))
        y = np.ones((3, 3))  # Different shape
        z = np.full((2, 2), 2)

        with self.assertRaises(AssertionError) as context:
            sample.add_arrays_kwargs(x=x, y=y, z=z)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    def test_shape_check_non_method_standalone_function_numpy(self):
        """Test that decorator correctly handles standalone NumPy function."""

        @shape_check("a", "b")
        def process(a, b):
            return a + b

        a = np.zeros((2, 2))
        b = np.ones((2, 2))

        result = process(a=a, b=b)
        expected = a + b
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_non_method_with_extra_arguments_numpy(self):
        """Test that decorator correctly ignores extra positional arguments beyond n."""

        @shape_check(2)
        def add_arrays(x, y, z):
            return x + y + z

        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_arrays(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_extra_positional_arguments_numpy(self):
        """Test that decorator works when n is less than the number of positional arguments."""

        @shape_check(2)
        def add(x, y, z):
            return x + y + z

        x = np.zeros((2, 2))
        y = np.ones((2, 2))
        z = np.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_shape_check_shape_attributes_present_but_not_matching_numpy(self):
        """Test that decorator correctly detects when shapes are not matching."""

        @shape_check("x", "y")
        def add(a, b):
            return a + b

        a = np.zeros((2, 2))
        b = np.ones((3, 3))  # Different shape

        with self.assertRaises(AssertionError) as context:
            add(x=a, y=b)
        self.assertIn("Shapes of the given arguments are not the same", str(context.exception))

    # -----------------------
    # Additional Tests for PyTorch Tensors
    # -----------------------

    def test_shape_check_non_method_with_extra_arguments_torch(self):
        """Test that decorator correctly ignores extra positional PyTorch tensor arguments beyond n."""

        @shape_check(2)
        def add_tensors(x, y, z):
            return x + y + z

        x = torch.zeros((2, 2))
        y = torch.ones((2, 2))
        z = torch.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_tensors(x, y, z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)

    def test_shape_check_extra_kwargs_not_checked_torch(self):
        """Test that decorator correctly ignores extra keyword PyTorch tensor arguments beyond specified keys."""

        @shape_check("x", "y")
        def add_tensors(**kwargs):
            return kwargs["x"] + kwargs["y"] + kwargs.get("z", 0)

        x = torch.zeros((2, 2))
        y = torch.ones((2, 2))
        z = torch.full((2, 2), 2)  # Modified: z's shape matches x and y

        result = add_tensors(x=x, y=y, z=z)
        expected = x + y + z
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected)


# Run the tests
if __name__ == "__main__":
    unittest.main()
