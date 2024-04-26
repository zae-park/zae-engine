import unittest

import torch

from zae_engine.models.foundations import resnet18, resnet34, resnet50, resnet101, resnet152
from zae_engine.models.foundations import seresnet18, seresnet34, seresnet50, seresnet101, seresnet152


class TestResnet(unittest.TestCase):
    test_sample = None

    @classmethod
    def setUpClass(cls) -> None:
        # cls.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))
        pass

    def setUp(self) -> None:
        self.test_sample = torch.randn((2, torch.randint(1, 16, size=[1]), 256, 256))

    def tearDown(self) -> None:
        pass

    def test_resnet18(self):
        model = resnet18()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_resnet34(self):
        model = resnet34()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_resnet50(self):
        model = resnet50()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_resnet101(self):
        model = resnet101()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_resnet152(self):
        model = resnet152()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_seresnet18(self):
        model = seresnet18()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_seresnet34(self):
        model = seresnet34()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_seresnet50(self):
        model = seresnet50()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_seresnet101(self):
        model = seresnet101()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)

    def test_seresnet152(self):
        model = seresnet152()
        output = model(self.test_sample)
        self.assertEqual(output.size(), 1000)


if __name__ == "__main__":
    unittest.main()
