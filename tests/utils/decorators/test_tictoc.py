import unittest
import time
from zae_engine.utils.decorators import tictoc


class TestTictocDecorator(unittest.TestCase):

    def test_tictoc_function(self):
        @tictoc
        def example_func():
            time.sleep(1)
            return "done"

        start = time.time()
        result = example_func()
        end = time.time()

        self.assertEqual(result, "done")
        self.assertTrue(end - start >= 1)

    def test_tictoc_method(self):
        class Example:
            @tictoc
            def example_method(self):
                time.sleep(1)
                return "done"

        example = Example()

        start = time.time()
        result = example.example_method()
        end = time.time()

        self.assertEqual(result, "done")
        self.assertTrue(end - start >= 1)


if __name__ == "__main__":
    unittest.main()
