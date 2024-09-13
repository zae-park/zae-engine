import unittest
import torch

from zae_engine.models.foundations import bert_base


class TestBert(unittest.TestCase):
    def setUp(self):
        # Define model hyperparameters
        self.dim_model = 768
        self.num_layers = 12
        self.sep_token_id = 102
        self.src_vocab_size = 30522
        self.max_len = 512

        # Load pre-trained model and tokenizer
        self.tokenizer, self.bert_model = bert_base(pretrained=True)

        # Generate input data
        self.input_ids = torch.randint(0, self.src_vocab_size, (32, self.max_len))  # batch_size=32, seq_len=max_len
        self.input_embeds = torch.randn(32, self.max_len, self.dim_model)  # Pre-embedded input

    def test_forward_with_input_ids(self):
        """Test forward pass with input_ids."""
        output = self.bert_model(input_sequence=self.input_ids)
        self.assertEqual(output.size(), (32, self.dim_model))  # Expecting pooled output from [CLS] token

    def test_forward_with_inputs_embeds(self):
        """Test forward pass with inputs_embeds."""
        output = self.bert_model(input_sequence=self.input_embeds)
        self.assertEqual(output.size(), (32, self.dim_model))  # Expecting full encoder output

    def test_sep_token_validation(self):
        """Test validation for more than two [SEP] tokens."""
        # Create a sequence with more than two [SEP] tokens
        input_ids_with_two_sep = torch.cat(
            [self.input_ids[:, : self.max_len - 2], torch.tensor([[self.sep_token_id, self.sep_token_id]] * 32)], dim=1
        )

        if max((self.input_ids == 102).sum(1)) > 1:
            with self.assertRaises(ValueError) as context:
                self.bert_model(input_sequence=input_ids_with_two_sep)

            self.assertIn("more than two [SEP] tokens", str(context.exception))

    def test_token_type_ids_generation(self):
        """Test token type IDs generation."""
        # Create a sequence with a single [SEP] token at the max length position
        input_ids_with_sep = torch.cat(
            [self.input_ids[:, : self.max_len - 1], torch.tensor([[self.sep_token_id]] * 32)], dim=1
        )

        # Generate token type IDs using Bert model
        token_type_ids = self.bert_model._generate_token_type_ids(input_ids_with_sep.tolist())

        # Check if token type IDs are correctly assigned
        # self.assertTrue(torch.all(token_type_ids[:, : self.max_len - 1] == 0))  # First segment should be 0
        self.assertTrue(torch.all(token_type_ids[:, self.max_len :] == 1))  # Second segment should be 1 after [SEP]


if __name__ == "__main__":
    unittest.main()
