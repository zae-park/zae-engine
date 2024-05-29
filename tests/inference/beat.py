import unittest

import numpy as np

from zae_engine.inference import beat


class TestBeatModel(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.zeros(2048000)

    def test_core(self):
        beat.core(self.x)


if __name__ == "__main__":
    unittest.main()
