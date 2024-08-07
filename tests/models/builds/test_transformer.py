import unittest
import torch

from zae_engine.models import TimeAwareTransformer


class TestTimeAwareTransformer(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 6
        self.model = TimeAwareTransformer(self.d_model, self.nhead, self.num_layers)
        self.batch_size = 16
        self.seq_len = 10
        self.event_vecs = torch.randn(self.seq_len, self.batch_size, 128)
        self.time_vecs = torch.randint(0, 512, (self.seq_len, self.batch_size))

    def test_forward(self):
        output = self.model(self.event_vecs, self.time_vecs)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))

    def test_variable_sequence_length(self):
        event_vecs = torch.randn(5, self.batch_size, 128)
        time_vecs = torch.randint(0, 512, (5, self.batch_size))
        output = self.model(event_vecs, time_vecs)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))

    def test_with_mask(self):
        mask = torch.ones(self.batch_size, self.seq_len).bool()
        output = self.model(self.event_vecs, self.time_vecs, mask)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))


if __name__ == "__main__":
    unittest.main()
