import unittest

import numpy as np

from zae_engine.inference import sec10


class TestBeatModel(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.zeros((32, 2500))

    def test_core(self):
        sec10.core(self.x, batch_size=16)


if __name__ == "__main__":
    unittest.main()
