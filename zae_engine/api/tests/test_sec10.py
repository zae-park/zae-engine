import unittest

import numpy as np

from zae_engine.api.sec10 import Sec10Collate


# class TestCollate(unittest.TestCase):
#
#     def get_attribute(self):
#         self.random_length = np.random.randint(9999) * np.random.randint(9999)
#         self.chunks = self.random_length // 2500
#         self.ex_signal = np.random.randn(self.random_length)
#         self.collate = Sec10Collate()
#
#     def setUp(self) -> None:
#         self.get_attribute()
#
#     def test_dtype(self):
#         chunk_out = self.collate.chunk({'x': self.ex_signal})
#         self.assertIsInstance(chunk_out, np.ndarray)
#
#         filter_out = self.collate.filtering(chunk_out)
#         self.assertIsInstance(filter_out, np.ndarray)
#
#         scale_out = self.collate.scaling(filter_out)
#         self.assertIsInstance(scale_out, np.ndarray)
#
#         collate_out = self.collate([{'x': self.ex_signal}])
#         self.assertIsInstance(collate_out, dict)
#
#     def test_chunk_size(self):
#         collate_out = self.collate([{'x': self.ex_signal}])
#         output_size = tuple(collate_out['x'].size())
#         self.assertTrue(output_size == (self.chunks, 1, 2500))


if __name__ == '__main__':
    unittest.main()
