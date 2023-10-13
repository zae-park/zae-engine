import unittest

import torch
import numpy as np

from zae_engine.operation import draw_confusion_matrix
from zae_engine.operation import print_confusion_matrix
from zae_engine.operation import MorphologicalLayer


class TestMorphology(unittest.TestCase):
    def setUp(self) -> None:
        self.zit_tensor = torch.tensor(np.array([1] * 80 + [0, 1] * 20 + [0] * 80), dtype=torch.float32).reshape(1, 1, -1)
        self.open_close = MorphologicalLayer('CO', [9, 9])
        self.close_open = MorphologicalLayer('OC', [9, 9])
        super().setUp()

    def test_morph(self):
        res1 = self.open_close(self.zit_tensor)
        res2 = self.close_open(self.zit_tensor)

        self.assertEqual(res1.shape, res2.shape)
        self.assertGreaterEqual(res1.sum(), res2.sum())


class TestDrawConfusion(unittest.TestCase):

    def setUp(self) -> None:
        self.num_classes = 30
        self.yhat_ex = np.random.randint(0, self.num_classes, 2000)
        self.y_ex = np.random.randint(0, self.num_classes, 2000)
        self.no_elements = np.array([])
        return super().setUp()

    def check_shape_and_classes(self):
        cm = draw_confusion_matrix(self.y_ex, self.yhat_ex, num_classes=self.num_classes)
        classes, _ = cm.shape

        self.assertEqual(classes, self.num_classes)
        self.assertEqual((self.num_classes, self.num_classes), cm.shape)

    def test_zero_num_classes(self):
        with self.assertRaises(Exception):
            draw_confusion_matrix(self.y_ex, self.yhat_ex, num_classes=0)
            # draw_confusion_matrix(no_ele_np, no_ele_np, num_classes=0)


class TestPrintConfusion(unittest.TestCase):

    def check_input_as_ndarray(self):
        with self.assertRaises(Exception):
            print_confusion_matrix(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))

