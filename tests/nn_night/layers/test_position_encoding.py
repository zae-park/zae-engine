import unittest
import torch
import torch.nn as nn
from position_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    RotaryPositionalEncoding,
    RelativePositionalEncoding,
    AdaptivePositionalEncoding,
)


class TestPositionEncodings(unittest.TestCase):

    def setUp(self):
        # Setting up common parameters for testing
        self.d_model = 128
        self.max_len = 512
        self.seq_len = 100
        self.batch_size = 10

    def test_sinusoidal_positional_encoding(self):
        pos_enc = SinusoidalPositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to added positional encoding."
        )

    def test_learnable_positional_encoding(self):
        pos_enc = LearnablePositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to added positional encoding."
        )

    def test_rotary_positional_encoding(self):
        pos_enc = RotaryPositionalEncoding(self.d_model)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to applied rotary positional encoding."
        )

    def test_relative_positional_encoding(self):
        pos_enc = RelativePositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to added positional encoding."
        )

    def test_adaptive_positional_encoding(self):
        pos_enc = AdaptivePositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        seq_lengths = torch.full((self.batch_size,), self.seq_len, dtype=torch.long)
        output = pos_enc(x, seq_lengths=seq_lengths)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to added positional encoding."
        )

    def test_adaptive_positional_encoding_with_variable_lengths(self):
        pos_enc = AdaptivePositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(self.batch_size, self.max_len, self.d_model)
        seq_lengths = torch.randint(1, self.max_len, (self.batch_size,), dtype=torch.long)
        output = pos_enc(x, seq_lengths=seq_lengths)

        self.assertEqual(output.shape, (self.batch_size, self.max_len, self.d_model))
        self.assertFalse(
            torch.equal(x, output), "Output should be different from input due to added positional encoding."
        )

        for i in range(self.batch_size):
            self.assertTrue(
                torch.equal(
                    output[i, : seq_lengths[i]], x[i, : seq_lengths[i]] + pos_enc.position_embeddings[: seq_lengths[i]]
                )
            )


if __name__ == "__main__":
    unittest.main()
