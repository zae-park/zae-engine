import unittest

import torch

from zae_engine.models import foundations


class TestResnet(unittest.TestCase):
    test_sample = None

    @classmethod
    def setUpClass(cls) -> None:
        # cls.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        pass

    def setUp(self) -> None:
        # self.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        self.test_sample = torch.randn((2, 3, 256, 256))

    def tearDown(self) -> None:
        pass

    def test_resnet18(self):
        model = foundations.resnet18()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_resnet34(self):
        model = foundations.resnet34()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_resnet50(self):
        model = foundations.resnet50()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_resnet101(self):
        model = foundations.resnet101()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_resnet152(self):
        model = foundations.resnet152()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_seresnet18(self):
        model = foundations.seresnet18()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_seresnet34(self):
        model = foundations.seresnet34()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_seresnet50(self):
        model = foundations.seresnet50()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_seresnet101(self):
        model = foundations.seresnet101()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_seresnet152(self):
        model = foundations.seresnet152()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_cbamresnet18(self):
        model = foundations.cbamresnet18()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_cbamresnet34(self):
        model = foundations.cbamresnet34()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_cbamresnet50(self):
        model = foundations.cbamresnet50()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_cbamresnet101(self):
        model = foundations.cbamresnet101()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))

    def test_cbamresnet152(self):
        model = foundations.cbamresnet152()
        output = model(self.test_sample)
        self.assertEqual(output.size(), torch.Size([2, 1000]))


if __name__ == "__main__":
    unittest.main()
