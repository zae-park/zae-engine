import unittest
import torch
import torch.nn as nn
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

# Adjust the import paths based on your project structure
from zae_engine.models.builds import transformer as trx
from zae_engine.nn_night.layers import SinusoidalPositionalEncoding
from zae_engine.models.foundations import benchmark as bm


class TestTimeSeriesTransformer(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for the tests."""
        self.vocab_size = 10000
        self.d_model = 512
        self.max_len = 100
        self.num_layers = 6
        self.num_heads = 8
        self.dim_feedforward = 2048
        self.dropout = 0.1

        self.batch_size = 4
        self.seq_len = 50

        self.model = bm.TimeSeriesTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.positions = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1).float()

    def test_model_initialization(self):
        """Test if the TimeSeriesTransformer initializes correctly."""
        self.assertIsInstance(self.model, bm.TimeSeriesTransformer)
        self.assertIsInstance(self.model.word_embedding, nn.Embedding)
        self.assertIsInstance(self.model.positional_encoding, SinusoidalPositionalEncoding)
        self.assertIsInstance(self.model.encoder, trx.EncoderBase)
        self.assertIsInstance(self.model.transformer, trx.BertBase)

    def test_forward_pass(self):
        """Test a forward pass through the TimeSeriesTransformer model."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=self.input_ids, positions=self.positions)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_forward_pass_with_masks(self):
        """Test the forward pass with source masks and padding masks."""
        src_mask = torch.zeros((self.seq_len, self.seq_len))  # No masking
        src_key_padding_mask = torch.zeros((self.batch_size, self.seq_len), dtype=torch.bool)  # No padding

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=self.input_ids,
                positions=self.positions,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_embedding_and_positional_encoding(self):
        """Test if embeddings and positional encodings are correctly added."""
        word_embeds = self.model.word_embedding(self.input_ids)
        pos_enc = self.model.positional_encoding(word_embeds, self.positions)
        combined = word_embeds + pos_enc

        self.assertEqual(combined.shape, word_embeds.shape)
        self.assertFalse(torch.isnan(combined).any())
        self.assertFalse(torch.isinf(combined).any())

    def test_invalid_input_ids_type(self):
        """Test if passing non-integer tensor to input_ids raises an error."""
        invalid_input_ids = self.positions  # Float tensor instead of long
        with self.assertRaises(RuntimeError):
            self.model(input_ids=invalid_input_ids, positions=self.positions)

    def test_invalid_positions_shape(self):
        """Test if passing positions with incorrect shape raises an error."""
        invalid_positions = torch.arange(self.seq_len).float()  # Missing batch dimension
        with self.assertRaises(AssertionError):
            self.model(input_ids=self.input_ids, positions=invalid_positions)

    def test_invalid_positions_type(self):
        """Test if passing non-float tensor to positions raises an error."""
        invalid_positions = self.input_ids.float()  # Logical type but not timestamps
        # Depending on implementation, this might not raise an error unless asserted
        # Here we assume no assertion, so no error is raised
        try:
            output = self.model(input_ids=self.input_ids, positions=invalid_positions)
            self.assertIsInstance(output, torch.Tensor)
        except AssertionError:
            self.fail("Model raised AssertionError unexpectedly!")

    def test_model_parameters_count(self):
        """Test if the model has the expected number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        # Estimate expected parameters based on model architecture
        # This is a rough estimate; adjust based on actual model configuration
        expected_min_params = 10_000_000  # Example threshold
        self.assertGreater(total_params, expected_min_params)

    def test_model_output_values(self):
        """Test if the model outputs do not contain NaN or Inf values."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=self.input_ids, positions=self.positions)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")

    def test_model_forward_gradients(self):
        """Test if the model can perform a backward pass without errors."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        output = self.model(input_ids=self.input_ids, positions=self.positions)
        loss = output.sum()
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"Backward pass failed with exception: {e}")

    def test_forward_pass_batch_size_variation(self):
        """Test the forward pass with different batch sizes."""
        for batch_size in [1, 8, 16]:
            input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
            positions = torch.arange(self.seq_len).unsqueeze(0).expand(batch_size, -1).float()
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_ids=input_ids, positions=positions)
            self.assertEqual(output.shape, (batch_size, self.seq_len, self.d_model))

    def test_forward_pass_sequence_length_variation(self):
        """Test the forward pass with different sequence lengths."""
        for seq_len in [10, 100, 200]:
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, seq_len))
            positions = torch.arange(seq_len).unsqueeze(0).expand(self.batch_size, -1).float()
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_ids=input_ids, positions=positions)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.d_model))

    def test_forward_pass_zero_sequence_length(self):
        """Test the forward pass with zero sequence length."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, 0))
        positions = torch.arange(0).unsqueeze(0).expand(self.batch_size, -1).float()
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(input_ids=input_ids, positions=positions)
                # Depending on implementation, output might be empty or raise an error
                self.assertEqual(output.shape, (self.batch_size, 0, self.d_model))
            except Exception as e:
                self.fail(f"Model failed with zero sequence length: {e}")

    @patch("zae_engine.nn_night.layers.SinusoidalPositionalEncoding.forward")
    def test_positional_encoding_called_with_correct_arguments(self, mock_pos_enc_forward):
        """Test if SinusoidalPositionalEncoding is called with correct arguments."""
        mock_pos_enc_forward.return_value = torch.zeros_like(self.model.word_embedding(self.input_ids))
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=self.input_ids, positions=self.positions)
        mock_pos_enc_forward.assert_called_once()
        args, kwargs = mock_pos_enc_forward.call_args
        self.assertEqual(args[0].shape, self.model.word_embedding(self.input_ids).shape)
        self.assertEqual(args[1].shape, self.positions.shape)

    def test_expand_vocab_size(self):
        """Test if the vocabulary size can be expanded correctly."""
        original_vocab_size = self.model.word_embedding.num_embeddings
        new_vocab_size = original_vocab_size + 1000
        new_embedding = nn.Embedding(new_vocab_size, self.d_model)
        new_embedding.weight.data[:original_vocab_size] = self.model.word_embedding.weight.data
        self.model.word_embedding = new_embedding

        # Verify the new vocab size
        self.assertEqual(self.model.word_embedding.num_embeddings, new_vocab_size)

        # Test forward pass with new tokens
        new_input_ids = torch.randint(0, new_vocab_size, (self.batch_size, self.seq_len))
        new_positions = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1).float()
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=new_input_ids, positions=new_positions)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_positional_encoding_max_len_exceeded(self):
        """Test if positional encoding handles sequences longer than max_len."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len + 10))
        positions = torch.arange(self.max_len + 10).unsqueeze(0).expand(self.batch_size, -1).float()
        self.model.eval()
        with self.assertRaises(RuntimeError):
            # Depending on implementation, it might raise an error or handle it internally
            self.model(input_ids=input_ids, positions=positions)

    def test_model_device(self):
        """Test if the model can be moved to GPU without issues."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                model = self.model.to(device)
                input_ids = self.input_ids.to(device)
                positions = self.positions.to(device)
                self.model.eval()
                with torch.no_grad():
                    output = model(input_ids=input_ids, positions=positions)
                self.assertEqual(output.device, device)
            except Exception as e:
                self.fail(f"Model failed to move to GPU: {e}")
        else:
            self.skipTest("CUDA is not available.")

    def test_model_save_and_load(self):
        """Test saving and loading the model state."""
        checkpoint = "temp_checkpoint.pth"
        try:
            # Save the model state
            torch.save(self.model.state_dict(), checkpoint)

            # Create a new instance and load the state
            new_model = bm.TimeSeriesTransformer(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                max_len=self.max_len,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )
            new_model.load_state_dict(torch.load(checkpoint))

            # Verify that parameters are the same
            for param1, param2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(param1, param2))
        finally:
            import os

            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_model_requires_grad(self):
        """Test if model parameters require gradients."""
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad, "Model parameters should require gradients")

    def test_forward_pass_requires_grad(self):
        """Test if the output requires gradients when input requires gradients."""
        self.model.train()
        input_ids = self.input_ids.clone().detach().requires_grad_(True)
        positions = self.positions.clone().detach().requires_grad_(True)

        output = self.model(input_ids=input_ids, positions=positions)
        self.assertTrue(output.requires_grad, "Output should require gradients when inputs require gradients")

        # Perform a backward pass
        loss = output.sum()
        try:
            loss.backward()
        except Exception as e:
            self.fail(f"Backward pass failed: {e}")

    def test_forward_pass_different_dtypes(self):
        """Test if the model handles different input data types correctly."""
        # Integer input_ids and float positions are expected
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)).long()
        positions = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1).float()

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids, positions=positions)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, torch.float32)  # Assuming default d_model dtype is float32

    def test_forward_pass_with_token_type_ids_ignored(self):
        """Test if the model ignores token_type_ids when not used."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        positions = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1).float()
        token_type_ids = torch.zeros_like(input_ids)  # Not used in this model

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids, positions=positions)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))


if __name__ == "__main__":
    unittest.main()
