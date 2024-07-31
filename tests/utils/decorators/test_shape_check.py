import unittest
import torch
from zae_engine.utils.decorators import shape_check


class TestShapeCheckDecorator(unittest.TestCase):

    def test_shape_check_function(self):
        @shape_check(2)
        def example_func(x, y):
            return x + y

        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self.assertEqual(example_func(x, y).tolist(), [5, 7, 9])

        y_invalid = torch.tensor([[4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError):
            example_func(x, y_invalid)

    def test_shape_check_method(self):
        class Example:
            @shape_check("x", "y")
            def example_method(self, x=None, y=None):
                return x + y

        example = Example()

        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self.assertEqual(example.example_method(x=x, y=y).tolist(), [5, 7, 9])

        y_invalid = torch.tensor([[4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError):
            example.example_method(x=x, y=y_invalid)

    def test_shape_check_failure(self):
        @shape_check(2)
        def example_func(x, y):
            return x + y

        x = torch.tensor([1, 2, 3])
        y_invalid = torch.tensor([[4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError):
            example_func(x, y_invalid)


if __name__ == "__main__":
    unittest.main()
