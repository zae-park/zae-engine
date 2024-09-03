import unittest
import torch
import random
from positional_encodings import (
    SinusoidalPositionalEncoding,
    TimestampPositionalEncoding,
    LearnablePositionalEncoding,
    RotaryPositionalEncoding,
    RelativePositionalEncoding,
    AdaptivePositionalEncoding,
)


class TestPositionalEncodings(unittest.TestCase):

    def setUp(self):
        # Generate random parameters
        self.batch_size = random.randint(1, 16)
        self.seq_len = random.randint(1, 100)
        self.d_model = random.randint(2, 128)  # Ensure d_model is even for rotary encoding

    def test_sinusoidal_positional_encoding(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoding_layer = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=self.seq_len)

        # Test without timestamps
        output = encoding_layer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

        # Test with timestamps
        timestamps = torch.randn(self.batch_size, self.seq_len)
        output_with_ts = encoding_layer(x, timestamps=timestamps)
        self.assertEqual(output_with_ts.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_timestamp_positional_encoding(self):
        timestamps = torch.randn(self.batch_size, self.seq_len)
        encoding_layer = TimestampPositionalEncoding(d_model=self.d_model)

        output = encoding_layer(timestamps)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_learnable_positional_encoding(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoding_layer = LearnablePositionalEncoding(d_model=self.d_model, max_len=self.seq_len)

        output = encoding_layer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_rotary_positional_encoding(self):
        # Ensure d_model is even for rotary encoding
        if self.d_model % 2 != 0:
            self.d_model += 1

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoding_layer = RotaryPositionalEncoding(d_model=self.d_model)

        output = encoding_layer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_relative_positional_encoding(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoding_layer = RelativePositionalEncoding(d_model=self.d_model, max_len=self.seq_len)

        output = encoding_layer(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_adaptive_positional_encoding(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        seq_lengths = torch.randint(1, self.seq_len + 1, (self.batch_size,))
        encoding_layer = AdaptivePositionalEncoding(d_model=self.d_model, max_len=self.seq_len)

        output = encoding_layer(x, seq_lengths=seq_lengths)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

        for i in range(self.batch_size):
            length = seq_lengths[i]
            self.assertTrue(torch.equal(output[i, length:], torch.zeros(self.seq_len - length, self.d_model)))


if __name__ == "__main__":
    unittest.main()
