import unittest
from unittest.mock import patch
import torch
import torch.nn as nn

from zae_engine.models.foundations import benchmark as bm  # Adjust import path as needed
from zae_engine.models.builds import transformer as trx
from zae_engine.nn_night.layers import SinusoidalPositionalEncoding


class TestTimeSeriesBert(unittest.TestCase):
    def setUp(self):
        """Set up model parameters for testing."""
        self.vocab_size = 10000
        self.d_model = 512
        self.max_len = 100
        self.num_layers = 6
        self.num_heads = 8
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.dim_pool = 512  # If you want to use the pooler

        # Initialize the model
        self.model = bm.TimeSeriesBert(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            dim_pool=self.dim_pool,
            batch_first=True,
        )

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.word_embedding, nn.Embedding)
        self.assertIsInstance(self.model.positional_encoding, SinusoidalPositionalEncoding)
        self.assertIsInstance(self.model.encoder, trx.EncoderBase)
        if self.dim_pool:
            self.assertIsInstance(self.model.pool_dense, nn.Linear)
            self.assertIsInstance(self.model.pool_activation, nn.Tanh)

    def test_forward_pass(self):
        """Test a forward pass through the model."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Forward pass
        output = self.model(input_ids=input_ids, positions=positions)

        # Check output shape
        if self.dim_pool:
            self.assertEqual(output.shape, (batch_size, self.dim_pool))
        else:
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_forward_pass_with_positions_none(self):
        """Test a forward pass when positions are None."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = None

        # Forward pass
        output = self.model(input_ids=input_ids, positions=positions)

        # Check output shape
        if self.dim_pool:
            self.assertEqual(output.shape, (batch_size, self.dim_pool))
        else:
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_embedding_with_positional_encoding(self):
        """Test if embeddings and positional encodings are correctly applied."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Access the embedding module
        word_embeds = self.model.word_embedding(input_ids)
        pos_embeds = self.model.positional_encoding(word_embeds, positions)
        combined_embeds = word_embeds + pos_embeds

        # Check embeddings shape
        self.assertEqual(combined_embeds.shape, (batch_size, seq_len, self.d_model))

    def test_forward_with_mask(self):
        """Test forward pass with source masks."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Create a source mask of shape [seq_len, seq_len]
        src_mask = torch.ones(seq_len, seq_len)  # No masking

        # Create a key padding mask of shape [batch_size, seq_len]
        src_key_padding_mask = torch.zeros(batch_size, seq_len).bool()  # No padding

        # Forward pass with masks
        output = self.model(
            input_ids=input_ids, positions=positions, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        # Check output shape
        if self.dim_pool:
            self.assertEqual(output.shape, (batch_size, self.dim_pool))
        else:
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_forward_with_padding(self):
        """Test forward pass with padding masks."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Create a padding mask where the last 10 tokens are padding
        src_key_padding_mask = torch.zeros(batch_size, seq_len).bool()
        src_key_padding_mask[:, -10:] = True  # Last 10 tokens are padding

        # Forward pass with padding masks
        output = self.model(input_ids=input_ids, positions=positions, src_key_padding_mask=src_key_padding_mask)

        # Check output shape
        if self.dim_pool:
            self.assertEqual(output.shape, (batch_size, self.dim_pool))
        else:
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    @patch("zae_engine.models.builds.transformer.EncoderBase.forward")
    def test_encoder_forward_call(self, mock_encoder_forward):
        """Test if the encoder's forward method is called correctly."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Define what the mock should return
        if self.dim_pool:
            mock_output = torch.randn(batch_size, self.d_model)
        else:
            mock_output = torch.randn(batch_size, seq_len, self.d_model)
        mock_encoder_forward.return_value = mock_output

        # Forward pass
        output = self.model(input_ids=input_ids, positions=positions)

        # Check if encoder's forward was called once
        mock_encoder_forward.assert_called_once()

        # Check if output matches mock output
        self.assertTrue(torch.equal(output, mock_output))

    def test_invalid_input_shape(self):
        """Test if the model raises an error with invalid input shapes."""
        batch_size = 4
        seq_len = 50
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        # Introduce a mismatch in batch size
        invalid_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size + 1, -1).float()

        with self.assertRaises(AssertionError):
            self.model(input_ids=input_ids, positions=invalid_positions)

    def test_invalid_token_ids(self):
        """Test if the model raises an error with token IDs out of range."""
        batch_size = 2
        seq_len = 10
        # Token IDs exceeding vocab_size
        input_ids = torch.randint(self.vocab_size, self.vocab_size + 100, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        with self.assertRaises(IndexError):
            self.model(input_ids=input_ids, positions=positions)

    def test_embedding_gradients(self):
        """Test if gradients are flowing through the embedding layer."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

        # Forward pass
        output = self.model(input_ids=input_ids, positions=positions)

        # Compute a simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check if gradients are not None
        self.assertIsNotNone(self.model.word_embedding.weight.grad)
        if self.dim_pool:
            self.assertIsNotNone(self.model.pool_dense.weight.grad)
        else:
            # If no pooler, check encoder gradients
            for param in self.model.encoder.parameters():
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
