import unittest

from zae_engine.data import example_ecg, example_mri


class Test_loader(unittest.TestCase):
    EX_10sec = None
    EX_beat = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.EX_10sec = example_ecg()
        cls.EX_beat = example_ecg(0)

    @classmethod
    def get_attribute(cls):
        example_10sec, example_beat = cls.EX_10sec, cls.EX_beat
        if example_10sec is None or example_beat is None:
            raise ValueError
        return example_10sec, example_beat

    def setUp(self) -> None:
        self.ex_10sec, self.ex_beat = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_example_ecg(self):
        # test for 10 sec example
        ex_10sec = example_ecg()
        self.assertEqual(len(ex_10sec), 2)                  # of returned elements check
        self.assertEqual(ex_10sec[0].shape, (2500, ))       # dimension check
        self.assertEqual(ex_10sec[1].shape, (2500, ))       # dimension check
        # test for 0'th beat example
        ex_beat = example_ecg(0)
        self.assertEqual(len(ex_beat), 3)                   # of returned elements check
        self.assertEqual(ex_beat[0].shape, (117,))          # dimension check
        self.assertEqual(type(ex_beat[1]), int)             # type check
        self.assertGreaterEqual(len(ex_beat[0]), ex_beat[1])    # check is r-peak in beat
        self.assertLessEqual(0, ex_beat[1])                     # check is r-peak in beat
        self.assertEqual(type(ex_beat[2]), str)             # check type of beat-type
        self.assertIn(ex_beat[2], ['N', 'A', 'V'])    # check beat-type is valid.

    def test_example_mri(self):
        res = example_mri()


if __name__ == '__main__':
    unittest.main()

