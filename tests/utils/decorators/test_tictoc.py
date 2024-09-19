import unittest
from unittest.mock import patch
import time
from typing import Callable, Any

# Import the optimized tictoc decorator
from zae_engine.utils.decorators import tictoc


class TestTictocDecorator(unittest.TestCase):
    """Unit tests for the tictoc decorator."""

    def test_function_timing(self):
        """Test that the decorator correctly measures the time of a standalone function."""

        @tictoc
        def sample_function():
            time.sleep(0.1)
            return "Function Completed"

        with patch("builtins.print") as mocked_print:
            result = sample_function()
            self.assertEqual(result, "Function Completed")
            mocked_print.assert_called_once()
            # Extract the elapsed time from the print statement
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (function 'sample_function'):", printed_output)

    def test_method_timing(self):
        """Test that the decorator correctly measures the time of a class method."""

        class SampleClass:
            @tictoc
            def sample_method(self):
                time.sleep(0.1)
                return "Method Completed"

        sample_instance = SampleClass()
        with patch("builtins.print") as mocked_print:
            result = sample_instance.sample_method()
            self.assertEqual(result, "Method Completed")
            mocked_print.assert_called_once()
            # Extract the elapsed time from the print statement
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (method 'sample_method'):", printed_output)

    def test_multiple_calls_function(self):
        """Test that the decorator works correctly on multiple calls to a standalone function."""

        @tictoc
        def sample_function(x):
            time.sleep(0.05)
            return x * 2

        with patch("builtins.print") as mocked_print:
            result1 = sample_function(2)
            result2 = sample_function(5)
            self.assertEqual(result1, 4)
            self.assertEqual(result2, 10)
            self.assertEqual(mocked_print.call_count, 2)
            # Check the print statements
            printed_output1 = mocked_print.call_args_list[0][0][0]
            printed_output2 = mocked_print.call_args_list[1][0][0]
            self.assertIn("Elapsed time [sec] (function 'sample_function'):", printed_output1)
            self.assertIn("Elapsed time [sec] (function 'sample_function'):", printed_output2)

    def test_multiple_calls_method(self):
        """Test that the decorator works correctly on multiple calls to a class method."""

        class SampleClass:
            @tictoc
            def sample_method(self, x):
                time.sleep(0.05)
                return x + 3

        sample_instance = SampleClass()
        with patch("builtins.print") as mocked_print:
            result1 = sample_instance.sample_method(2)
            result2 = sample_instance.sample_method(5)
            self.assertEqual(result1, 5)
            self.assertEqual(result2, 8)
            self.assertEqual(mocked_print.call_count, 2)
            # Check the print statements
            printed_output1 = mocked_print.call_args_list[0][0][0]
            printed_output2 = mocked_print.call_args_list[1][0][0]
            self.assertIn("Elapsed time [sec] (method 'sample_method'):", printed_output1)
            self.assertIn("Elapsed time [sec] (method 'sample_method'):", printed_output2)

    def test_no_sleep_function(self):
        """Test that the decorator works correctly on a function that does not sleep."""

        @tictoc
        def quick_function():
            return "Quick Function"

        with patch("builtins.print") as mocked_print:
            result = quick_function()
            self.assertEqual(result, "Quick Function")
            mocked_print.assert_called_once()
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (function 'quick_function'):", printed_output)

    def test_no_sleep_method(self):
        """Test that the decorator works correctly on a method that does not sleep."""

        class SampleClass:
            @tictoc
            def quick_method(self):
                return "Quick Method"

        sample_instance = SampleClass()
        with patch("builtins.print") as mocked_print:
            result = sample_instance.quick_method()
            self.assertEqual(result, "Quick Method")
            mocked_print.assert_called_once()
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (method 'quick_method'):", printed_output)

    def test_error_in_function(self):
        """Test that the decorator does not suppress errors in the decorated function."""

        @tictoc
        def error_function():
            time.sleep(0.05)
            raise ValueError("Intentional Error")

        with patch("builtins.print") as mocked_print:
            with self.assertRaises(ValueError) as context:
                error_function()
            self.assertEqual(str(context.exception), "Intentional Error")
            mocked_print.assert_called_once()
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (function 'error_function'):", printed_output)

    def test_error_in_method(self):
        """Test that the decorator does not suppress errors in the decorated method."""

        class SampleClass:
            @tictoc
            def error_method(self):
                time.sleep(0.05)
                raise ValueError("Intentional Method Error")

        sample_instance = SampleClass()
        with patch("builtins.print") as mocked_print:
            with self.assertRaises(ValueError) as context:
                sample_instance.error_method()
            self.assertEqual(str(context.exception), "Intentional Method Error")
            mocked_print.assert_called_once()
            printed_output = mocked_print.call_args[0][0]
            self.assertIn("Elapsed time [sec] (method 'error_method'):", printed_output)


if __name__ == "__main__":
    unittest.main()
