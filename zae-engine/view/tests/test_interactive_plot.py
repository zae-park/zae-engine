import unittest

import numpy as np

from .._interactive_plot import seg_plot


class TestSegPlot(unittest.TestCase):
    x = np.sin(np.linspace(0, 4 * np.pi, 2500))
    y = np.array(x > 0.5, dtype=int)
    p = y.copy()
    p[1000:1500] = 2
    rp = np.arange(10, 1210, 100)
    class_name = ['base', 'normal', 'miss']

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def get_attribute(cls):
        return {'x': cls.x, 'y': cls.y, 'p': cls.p, 'rp': cls.rp, 'class_name': cls.class_name}

    def setUp(self) -> None:
        self.attr_dict = self.get_attribute()

    def test_seg_plot(self):
        x, y, p, rp, class_name = self.attr_dict.values()
        seg_plot(x)
        seg_plot(x, true=y)
        seg_plot(x, pred=p)
        seg_plot(x, rpeak=rp)
        seg_plot(x, true=y, pred=p)
        seg_plot(x, true=y, rpeak=rp, save_path=None)
        seg_plot(x, rpeak=rp, class_names=list(range(6)))
        seg_plot(x, true=y, pred=p, rpeak=rp, minmax=True, class_names=list(range(6)), save_path=None)
        seg_plot(x, true=y, rpeak=rp, minmax=True, save_path=None)
        seg_plot(x, true=y, rpeak=rp, minmax=True, class_names=list(range(6)), save_path=None)
        seg_plot(x, true=y, pred=p, rpeak=rp, minmax=True, class_names=list(range(6)), save_path=None)
