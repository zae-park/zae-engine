import unittest
import torch

from zae_engine.loss import ArcFaceLoss


class TestArcFaceLoss(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.out_features = 10
        self.model = ArcFaceLoss(self.in_features, self.out_features)
        self.features = torch.randn(32, self.in_features)
        self.labels = torch.randint(0, self.out_features, (32,))

    def test_forward(self):
        logits = self.model(self.features, self.labels)
        self.assertEqual(logits.size(), (32, self.out_features))

    def test_different_s_m(self):
        model = ArcFaceLoss(self.in_features, self.out_features, s=64.0, m=0.25)
        logits = model(self.features, self.labels)
        self.assertEqual(logits.size(), (32, self.out_features))

    def test_different_input_size(self):
        features = torch.randn(16, self.in_features)
        labels = torch.randint(0, self.out_features, (16,))
        logits = self.model(features, labels)
        self.assertEqual(logits.size(), (16, self.out_features))


if __name__ == "__main__":
    unittest.main()
