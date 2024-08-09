import unittest
import torch

from zae_engine.models import TimeAwareTransformer


class TestTimeAwareTransformer(unittest.TestCase):
    def setUp(self):
        self.d_head = 32
        self.d_model = 128
        self.n_head = 8
        self.num_layers = 6
        self.model = TimeAwareTransformer(
            d_head=self.d_head, d_model=self.d_model, n_head=self.n_head, num_layers=self.num_layers
        )
        self.batch_size = 16
        self.seq_len = 10
        self.event_vecs = torch.randn(self.batch_size, self.seq_len, self.d_head)
        self.time_vecs = torch.randint(0, 512, (self.batch_size, self.seq_len))
        print()

    def test_forward(self):
        output = self.model(self.event_vecs, self.time_vecs)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))

    def test_variable_sequence_length(self):
        event_vecs = torch.randn(self.batch_size, 5, self.d_head)
        time_vecs = torch.randint(0, 512, (self.batch_size, 5))
        output = self.model(event_vecs, time_vecs)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))

    def test_with_mask(self):
        mask = torch.ones(self.batch_size, self.seq_len).bool()
        output = self.model(self.event_vecs, self.time_vecs, mask)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))


if __name__ == "__main__":
    unittest.main()
