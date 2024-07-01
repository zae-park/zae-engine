import unittest
import torch

from zae_engine.models import UserIdModel


class TestUserIdentificationModel(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 6
        self.num_classes = 1000
        self.model = UserIdentificationModel(self.d_model, self.nhead, self.num_layers, self.num_classes)
        self.batch_size = 16
        self.seq_len = 10
        self.event_vecs = torch.randn(self.seq_len, self.batch_size, 128)
        self.time_vecs = torch.randint(0, 512, (self.seq_len, self.batch_size))
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_forward(self):
        logits, features = self.model(self.event_vecs, self.time_vecs, self.labels)
        self.assertEqual(logits.size(), (self.batch_size, self.num_classes))
        self.assertEqual(features.size(), (self.batch_size, self.d_model))

    def test_expand_classes(self):
        new_num_classes = 1100
        self.model.expand_classes(new_num_classes)
        self.assertEqual(self.model.num_classes, new_num_classes)
        self.assertEqual(self.model.arcface.weight.size(), (new_num_classes, self.d_model))

    def test_forward_after_expand_classes(self):
        new_num_classes = 1100
        self.model.expand_classes(new_num_classes)
        logits, features = self.model(self.event_vecs, self.time_vecs, self.labels)
        self.assertEqual(logits.size(), (self.batch_size, new_num_classes))
        self.assertEqual(features.size(), (self.batch_size, self.d_model))

    def test_variable_sequence_length(self):
        event_vecs = torch.randn(5, self.batch_size, 128)
        time_vecs = torch.randint(0, 512, (5, self.batch_size))
        logits, features = self.model(event_vecs, time_vecs, self.labels)
        self.assertEqual(logits.size(), (self.batch_size, self.num_classes))
        self.assertEqual(features.size(), (self.batch_size, self.d_model))


if __name__ == "__main__":
    unittest.main()
