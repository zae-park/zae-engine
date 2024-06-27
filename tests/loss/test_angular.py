import unittest
import torch

from zae_engine.loss import ArcFaceLoss


class TestArcFaceLoss(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.out_features = 10
        self.model = ArcFaceLoss(self.in_features, self.out_features).cuda()
        self.features = torch.randn(32, self.in_features).cuda()
        self.labels = torch.randint(0, self.out_features, (32,)).cuda()

    def test_forward(self):
        logits = self.model(self.features, self.labels)
        self.assertEqual(logits.size(), (32, self.out_features))


if __name__ == "__main__":
    unittest.main()
