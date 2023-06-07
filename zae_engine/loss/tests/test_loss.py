import unittest

import numpy as np
import torch

from .. import _loss


class Test_loader(unittest.TestCase):
    SEED = 100
    num_class = 10
    dim = 1000
    torch.manual_seed(SEED)
    cls_logit = torch.randn((100, num_class))
    cls_proba = torch.softmax(cls_logit, dim=-1)
    cls_p_label = torch.zeros_like(cls_proba)

    seg_logit = torch.randn((100, num_class, dim))
    seg_proba = torch.softmax(seg_logit, dim=1)
    seg_p_label = torch.zeros_like(seg_proba)

    onoff_predict = torch.tensor([[11, 21], [28, 44], [49, 57], [72, 78]])
    onoff_label = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def get_attribute(cls):
        return {'cls': (cls.cls_logit, cls.cls_proba, cls.cls_p_label),
                'seg': (cls.seg_logit, cls.seg_proba, cls.seg_p_label),
                'onoff': (cls.onoff_predict, cls.onoff_label)}

    def setUp(self) -> None:
        self.attr_dict = self.get_attribute()

    def tearDown(self) -> None:
        pass

    def test_cross_entropy(self):
        self.cls_tuple = self.attr_dict['cls']
        self.seg_tuple = self.attr_dict['seg']
        with self.assertRaises(TypeError):
            np_loss = _loss.cross_entropy(self.cls_tuple[0].detach().numpy(), self.cls_tuple[-1].detach().numpy())
        loss = _loss.cross_entropy(self.cls_tuple[0], self.cls_tuple[-1])
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.size().numel(), 1)
        self.assertAlmostEqual(float(loss), 0.80846738, places=4)

        with self.assertRaises(TypeError):
            np_loss = _loss.cross_entropy(self.seg_tuple[0].detach().numpy(), self.seg_tuple[-1].detach().numpy())
        loss = _loss.cross_entropy(self.seg_tuple[0], self.seg_tuple[-1])
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.size().numel(), 1)
        self.assertAlmostEqual(float(loss), 0.80567473, places=4)

    def test_GIoU(self):
        self.onoff_tuple = self.attr_dict['onoff']
        with self.assertRaises(AssertionError):
            float_loss = _loss.GIoU(self.onoff_tuple[0].float(), self.onoff_tuple[-1].float())
        loss1 = _loss.GIoU(self.onoff_tuple[0].int(), self.onoff_tuple[1].int())
        loss2 = _loss.GIoU(self.onoff_tuple[1].int(), self.onoff_tuple[1].int())
        self.assertGreaterEqual(loss1, loss2)
        self.assertEqual(loss2, 0)

    def test_IoU(self):
        self.onoff_tuple = self.attr_dict['onoff']
        with self.assertRaises(AssertionError):
            float_loss = _loss.IoU(self.onoff_tuple[0].float(), self.onoff_tuple[-1].float())
        loss1 = _loss.IoU(self.onoff_tuple[0].int(), self.onoff_tuple[1].int())
        loss2 = _loss.IoU(self.onoff_tuple[1].int(), self.onoff_tuple[1].int())
        self.assertGreaterEqual(loss1, loss2)
        self.assertEqual(loss2, 0)

    def test_mIoU(self):
        self.seg_tuple = self.attr_dict['seg']
        with self.assertRaises(AssertionError):
            float_loss = _loss.mIoU(self.seg_tuple[1].argmax(1), self.seg_tuple[-1])

        loss1 = _loss.mIoU(self.seg_tuple[1].argmax(1), self.seg_tuple[-1].argmax(1))
        loss2 = _loss.mIoU(self.seg_tuple[1].argmax(1), torch.ones_like(self.seg_tuple[-1].argmax(1), dtype=torch.int))
        self.assertLessEqual(loss1, loss2)


if __name__ == '__main__':
    unittest.main()
