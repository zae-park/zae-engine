import unittest

import numpy as np

from zae_engine.api import beat
from zae_engine.data import load_example


# class Test_loader(unittest.TestCase):
#     EX_beat = None
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.EX_beat, _ = load_example()
#
#     @classmethod
#     def get_attribute(cls):
#         return cls.EX_beat
#
#     def setUp(self) -> None:
#         self.ex_beat = self.get_attribute()
#
#     def tearDown(self) -> None:
#         pass
#
#     def test_load_example(self):
#         result = beat.core(np.concatenate([self.ex_beat] * 100))
#         self.assertIsInstance(result, list)
#         self.assertEqual(len(result), 1300)


if __name__ == '__main__':
    unittest.main()

