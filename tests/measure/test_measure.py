import unittest

import numpy as np

from zae_engine import measure as _measure


class Test_loader(unittest.TestCase):
    def setUp(self) -> None:
        self.x_ex = np.array([0] * 500 + [1] * 500 + [0] * 1500)
        self.y_ex = np.array([0] * 600 + [1] * 400 + [0] * 1500)
        super().setUp()

    def test_accuracy(self):
        self.assertEqual(_measure.accuracy(self.y_ex, self.x_ex), 24 / 25)

    def test_bijective(self):
        bi = _measure.BijectiveMetric(self.x_ex, self.y_ex, num_class=2)
        bi.summary()

    def test_iec(self):
        d0 = {'sample': [100, 200, 300, 400, 500], 'rhythm': ['(N', '(N', '(AF', '(AF', '(N']}      # 300~500
        d1 = {'sample': [100, 200, 300, 400, 500], 'rhythm': ['(AF', '(N', '(AF', '(AF', '(N']}     # 100~200, 300~500
        d2 = {'sample': [100, 200, 300, 400], 'rhythm': ['(AF', '(AF', '(AF', '(AF']}               # 100~
        d3 = {'sample': [300, 500, 300, 400], 'rhythm': ['(AF', '(AF', '(AF', '(AF']}               # not sorted
        d4 = {'sample': [600, 700, 800, 900], 'rhythm': ['(N', '(N', '(N', '(AF']}                  # 900~
        self.assertEqual(_measure.iec_60601(d0, d1, 1000, '(AF')[0], _measure.iec_60601(d1, d0, 1000, '(AF')[1])
        with self.assertRaises(AssertionError) as err:
            _measure.iec_60601(d2, d3, 1000, '(AF')
            self.assertTrue(isinstance(err.exception, AssertionError))
        with self.assertRaises(KeyError) as err:
            _measure.iec_60601(d2, {'a': [1, 2, 3], 'b': [5, 6, 7]}, 1000, '(AF')
            self.assertTrue(isinstance(err.exception, KeyError))
        self.assertEqual(_measure.iec_60601(d2, d4, 800, '(NN')[0], 0)

    def test_cpsc(self):
        true = [10, 15, 30, 52, 69, 180]
        predict = [9, 16, 36, 52, 90, 140]
        predict2 = [10, 16, 36, 52, 90, 140]
        predict3 = [6, 16, 36, 52, 90, 140]
        self.assertEqual(_measure.cpsc2021(true, predict2), _measure.cpsc2021(true, predict))
        self.assertGreater(_measure.cpsc2021(true, predict2), _measure.cpsc2021(true, predict3))



if __name__ == '__main__':
    unittest.main()
